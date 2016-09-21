#include "PostStat.h"

#include "TrainingNodes.h"

namespace Microsoft { namespace MSR{ namespace CNTK {

template <class ElemType>
void PostStatistics<ElemType>::PostBatchNormalStatistics(IDataReader * dataReader, const vector<wstring>& evalNodeNames, 
    const wstring newModelPath, const size_t mbSize, const int iters)
{
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::training);

        // determine nodes to evaluate
        std::vector<ComputationNodeBasePtr> evalNodes;

        set<ComputationNodeBasePtr> criteriaLogged; // (keeps track ot duplicates to avoid we don't double-log critera)
        if (evalNodeNames.size() == 0)
        {
            fprintf(stderr, "evalNodeNames are not specified, using all the default evalnodes and training criterion nodes.\n");
            if (m_net->EvaluationNodes().empty() && m_net->FinalCriterionNodes().empty())
                InvalidArgument("There is no default evaluation node or training criterion specified in the network.");

            for (const auto& node : m_net->EvaluationNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);

            for (const auto& node : m_net->FinalCriterionNodes())
                if (criteriaLogged.insert(node).second)
                    evalNodes.push_back(node);
        }
        else
        {
            for (int i = 0; i < evalNodeNames.size(); i++)
            {
                const auto& node = m_net->GetNodeFromName(evalNodeNames[i]);
                if (!criteriaLogged.insert(node).second)
                    continue;
                if (node->GetSampleLayout().GetNumElements() != 1)
                    InvalidArgument("Criterion nodes to evaluate must have dimension 1x1.");
                evalNodes.push_back(node);
            }
        }

        // all batch normalization nodes should be marked and reset the mean and variance to initial state
        std::vector<ComputationNodeBasePtr> batchNormalNodes;
        std::set<ComputationNodeBasePtr> batchNormalLogged; // (avoid double record of batch normalization nodes)
        for (auto& evalNode : evalNodes)
        {
            for (auto& node : m_net->GetEvalOrder(evalNode))
            {
                shared_ptr<BatchNormalizationNode<ElemType>> batchNormalNode =
                    dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(node);
                if (batchNormalNode)
                {
                    if (batchNormalLogged.insert(node).second)
                    {
                        batchNormalNode->ResetStatisiticsState();
                        batchNormalNode->SetNormalizationTimeConstants(-1, batchNormalNode->NormalizationTimeConstant(),
                            0, batchNormalNode->BlendTimeConstant());
                        m_net->FormEvalOrder(node);
                        m_net->FormNestedNetwork(node);
                        batchNormalNodes.push_back(node);
                    }
                }
            }
        }

        // allocate memory for forward computation
        m_net->AllocateAllMatrices(evalNodes, batchNormalNodes, nullptr);

        // prepare features and labels
        auto& featureNodes = m_net->FeatureNodes();
        auto& labelNodes = m_net->LabelNodes();

        StreamMinibatchInputs inputMatrices;
        for (auto& node : featureNodes)
            inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());
        for (auto& node : labelNodes)
            inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());

        bool useParallelTrain = (m_mpi != nullptr);
        bool useDistributedMBReading = useParallelTrain && m_enableDistributedMBReading && dataReader->SupportsDistributedMBRead();
        if (useDistributedMBReading)
            dataReader->StartDistributedMinibatchLoop(mbSize, 0, m_mpi->CurrentNodeRank(), m_mpi->NumNodesInUse());
        else
            dataReader->StartMinibatchLoop(mbSize, 0);

        // Passing in two empty node lists so the dispatcher can work for the evalNodes.
        std::list<ComputationNodeBasePtr> learnableNodes;


        m_net->StartEvaluateMinibatchLoop(batchNormalNodes);

        // Push all batch normalization mean and std into learn params values for mpi update
        std::vector<Matrix<ElemType>*> learnParamsValues(2, nullptr);

        bool noMoreSamplesToProcess = false;
        for (auto& node : batchNormalNodes)
        {
            shared_ptr<BatchNormalizationNode<ElemType>> batchNormalNode =
                static_pointer_cast<BatchNormalizationNode<ElemType>>(node);
            size_t actualMBSize = 0;

            LOGPRINTF(stderr, "Start evaluating: %ls\n", batchNormalNode->GetName().c_str());

            // Post batch normal iters
            for (int iter = 0; iter < iters; iter++)
            {
                bool wasDataRead = DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(*dataReader, m_net,
                    nullptr, useDistributedMBReading, useParallelTrain, inputMatrices, actualMBSize, m_mpi);

                if (!wasDataRead && (!useDistributedMBReading || noMoreSamplesToProcess))
                    break;

                // TODO should handle it, since post BN exist no samples in iters
                if (!wasDataRead)
                    actualMBSize = 0;

                // Batch Normalization Evaluate don't need to support subMinibatches
                ComputationNetwork::BumpEvalTimeStamp(featureNodes);
                ComputationNetwork::BumpEvalTimeStamp(labelNodes);

                m_net->ForwardProp(node);
                dataReader->DataEnd();
            }

            batchNormalNode->FreezeParameters();

            // Sync during or after all iters of a BN node are equivalent
            if (useParallelTrain)
            {
                if (m_gradHeader == nullptr)
                {
                    m_gradHeader.reset(DistGradHeader::Create(evalNodes.size()), [](DistGradHeader* ptr)
                    {
                        DistGradHeader::Destroy(ptr);
                    });
                }
                SimpleDistGradAggregator<ElemType> distGradAgg(m_mpi, false /*useAsyncAggregation*/, 0 /*syncStatsTrace*/);

                auto runMeanParameterPtr = node->GetInputs()[3];
                auto runStdParameterPtr = node->GetInputs()[4];

                shared_ptr<ComputationNode<ElemType>> runMeanNode = static_pointer_cast<ComputationNode<ElemType>>(runMeanParameterPtr);
                shared_ptr<ComputationNode<ElemType>> runStdNode = static_pointer_cast<ComputationNode<ElemType>>(runStdParameterPtr);

                learnParamsValues[0] = &(runMeanNode->Value());
                learnParamsValues[1] = &(runStdNode->Value());

                m_gradHeader->numSamples = actualMBSize ? 1 : actualMBSize;
                distGradAgg.AggregateGradients(learnParamsValues, m_gradHeader.get(), 0);

                for (auto& parameter : learnParamsValues)
                {
                    (*parameter) /= (ElemType)m_mpi->NumNodesInUse();
                }
            }
        }

        // Save Model
        if (!useParallelTrain || m_mpi->CurrentNodeRank() == m_mpi->MainNodeRank())
            m_net->Save(newModelPath);

        return;
    }
}

template class PostStatistics<float>;
template class PostStatistics<double>;

}}}
