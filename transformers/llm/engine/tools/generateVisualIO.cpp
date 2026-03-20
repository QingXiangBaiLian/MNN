#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"
#include <limits>

static void saveInputOutputs(
    const MNN::Express::Module::Info* info,
    std::vector<MNN::Express::VARP> inputs,
    std::vector<MNN::Express::VARP> outputs,
    const std::string& outputDir,
    int index)
{
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i = 0; i < info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i = 0; i < info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, std::to_string(index));
    if (!(MNNCreateDir(subDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }
    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generated %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

// 生成对应 visual.mnn 的测试输入
static void createInputsForVisual(int chunkSize, std::vector<MNN::Express::VARP>& inputs) {
    // patches: [chunkSize, 1536]
    {
        MNN::Express::VARP patches = MNN::Express::_Input({chunkSize, 1536}, MNN::Express::NCHW, halide_type_of<float>());
        float* patchData = patches->writeMap<float>();
        for (int i = 0; i < chunkSize * 1536; ++i) {
            patchData[i] = (float)(rand()) / RAND_MAX;
        }
        inputs.push_back(patches);
    }

    // position_ids: [2, chunkSize]
    {
        MNN::Express::VARP positionIds = MNN::Express::_Input({2, chunkSize}, MNN::Express::NCHW, halide_type_of<int>());
        int* posData = positionIds->writeMap<int>();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < chunkSize; ++j) {
                posData[i * chunkSize + j] = j;
            }
        }
        inputs.push_back(positionIds);
    }

    // attention_mask: [1, chunkSize, chunkSize]
    {
        MNN::Express::VARP attMask = MNN::Express::_Input({1, chunkSize, chunkSize}, MNN::Express::NCHW, halide_type_of<float>());
        float* maskData = attMask->writeMap<float>();
        for (int i = 0; i < chunkSize; ++i) {
            for (int j = 0; j < chunkSize; ++j) {
                maskData[i * chunkSize + j] = (j > i) ? std::numeric_limits<float>::lowest() : 0.0f;
            }
        }
        inputs.push_back(attMask);
    }

    // idx_tensor: [4, chunkSize]
    {
        MNN::Express::VARP idxTensor = MNN::Express::_Input({4, chunkSize}, MNN::Express::NCHW, halide_type_of<int>());
        int* idxData = idxTensor->writeMap<int>();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < chunkSize; ++j) {
                idxData[i * chunkSize + j] = j;
            }
        }
        inputs.push_back(idxTensor);
    }

    // weight_tensor: [4, chunkSize]
    {
        MNN::Express::VARP weightTensor = MNN::Express::_Input({4, chunkSize}, MNN::Express::NCHW, halide_type_of<float>());
        float* weightData = weightTensor->writeMap<float>();
        for (int i = 0; i < 4 * chunkSize; ++i) {
            weightData[i] = (float)(rand()) / RAND_MAX - 0.5f;
        }
        inputs.push_back(weightTensor);
    }
}

// 生成视觉模型的测试输入输出
static void generateForVisual(const std::string& modelPath, const std::string& outputDir, int chunkSize) {
    // setup input/output names
    std::vector<std::string> inputNames = {"patches", "position_ids", "attention_mask", "idx_tensor", "weight_tensor"};
    std::vector<std::string> outputNames = {"image_embeds"}; // 按你的模型实际输出名调整

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;

    auto rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));

    MNN_PRINT("Loading visual model with %d inputs:\n", inputNames.size());
    for (const auto& name : inputNames) {
        MNN_PRINT("  %s\n", name.c_str());
    }

    auto net = std::shared_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr),
        MNN::Express::Module::destroy);
    if (!net) {
        MNN_ERROR("Failed to load visual mnn model: %s\n", modelPath.c_str());
        return;
    }
    MNN_PRINT("Visual model loaded successfully.\n");

    auto info = net->getInfo();
    MNN_PRINT("Model has %d inputs, %d outputs\n", info->inputNames.size(), info->outputNames.size());

    // 测试 chunkSize
    {
        std::vector<MNN::Express::VARP> inputs, outputs;
        createInputsForVisual(chunkSize, inputs);
        MNN_PRINT("Run forward for chunk size %d...\n", chunkSize);
        outputs = net->onForward(inputs);
        if (!outputs.empty()) {
            saveInputOutputs(info, inputs, outputs, outputDir, chunkSize);
            MNN_PRINT("Successfully saved outputs for chunk size %d\n", chunkSize);
        } else {
            MNN_ERROR("No outputs generated for chunk size %d\n", chunkSize);
        }
    }
    //// 测试 chunkSize=1
    //{
    //    std::vector<MNN::Express::VARP> inputs, outputs;
    //    createInputsForVisual(chunkSize, inputs);
    //    MNN_PRINT("Run forward for chunk size 1...\n");
    //    outputs = net->onForward(inputs);
    //    if (!outputs.empty()) {
    //        saveInputOutputs(info, inputs, outputs, outputDir, 1);
    //        MNN_PRINT("Successfully saved outputs for chunk size 1\n");
    //    } else {
    //        MNN_ERROR("No outputs generated for chunk size 1\n");
    //    }
    //}
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateVisualIO visual_model.mnn output_directory [chunk_size]\n");
        MNN_PRINT("   e.g. ./generateVisualIO ./visual.mnn ./output 128\n");
        return 1;
    }

    srand(time(NULL));
    int chunkSize = 676;
    if (argc >= 4) {
        chunkSize = atoi(argv[3]);
    }
    MNN_PRINT("chunkSize=%d\n", chunkSize);

    std::string modelPath = argv[1];
    std::string outputDir = argv[2];

    if (!(MNNCreateDir(outputDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    generateForVisual(modelPath, outputDir, chunkSize);

    return 0;
}
