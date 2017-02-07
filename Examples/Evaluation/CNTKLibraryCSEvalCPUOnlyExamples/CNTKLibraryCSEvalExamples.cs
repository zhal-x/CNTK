//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCSEvalExamples.cs -- Examples for using CNTK Library C# Eval API.
//

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    public class CNTKLibraryManagedExamples
    {
        // 
        // The example shows
        // - how to load model.
        // - how to prepare input data for a single sample.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in dense format.
        //
        public static void EvaluationSingleImage(DeviceDescriptor device)
        {
            const string outputName = "Plus2060";
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            Function modelFunc = Function.LoadModel("z.model", device);

            // Get output variable based on name
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
 
            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Single();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("Evaluate single image");

            // Image preprocessing to match input requirements of the model.
            Bitmap bmp = new Bitmap(Bitmap.FromFile("00000.png"));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);

            PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data for a batch of samples.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in dense format.
        //
        public static void EvaluationBatchOfImages(DeviceDescriptor device)
        {
            const string outputName = "Plus2060";
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            Function modelFunc = Function.LoadModel("z.model", device);

            // Get output variable based on name
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Single();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("\nEvaluate batch of images");

            Bitmap bmp, resized;
            List<float> resizedCHW;

            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png" };
            var seqData = new List<float>();
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                bmp = new Bitmap(Bitmap.FromFile(fileList[sampleIndex]));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer.
                seqData.AddRange(resizedCHW);
            }

            // Create Value for the batch data.
            inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);

            // Create input data map.
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Evaluate the model against the batch input
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Retrieve the evaluation result.
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);

            // Output result
            PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data as sequence.
        // - how to retrieve input and output variables by name.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in the one-hot vector format.
        //
        public static void EvaluationSingleSequenceUsingOneHot(DeviceDescriptor device)
        {
            //var vocabToIndex = buildVocabIndex("ATIS.vocab");
            //var indexToVocab = buildInvVocabIndex("ATIS.label");
            var vocabToIndex = buildVocabIndex(@"C:\repos\cntk\Examples\LanguageUnderstanding\ATIS\Data\ATIS.vocab");
            var indexToVocab = buildInvVocabIndex(@"C:\repos\cntk\Examples\LanguageUnderstanding\ATIS\Data\ATIS.label");

            Function myFunc = Function.LoadModel(@"C:\CNTKMisc\DataBackup\ATIS\work\ATIS.slot.lstm", device);

            Console.WriteLine("Evaluate single sequence using one-hot vector");

            // Get input variable 
            var cwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresCW")).Single();
            var nwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresNW")).Single();
            var pwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresPW")).Single();

            uint vocabSize = cwVar.Shape.TotalSize;

            // Use case 1: Evalaute a single sequence using OneHot vector as input.
            var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
            // Build input data for one sequence 
            var wordIds = new List<uint>();
            var cwSeqData = new List<uint>();
            var nwSeqData = new List<uint>();
            var pwSeqData = new List<uint>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlag = true;
            // Get the index of each word in the sentence.
            string[] substring = inputSentence.Split(' ');
            foreach (var str in substring)
            {
                // Get the index of the word
                var index = vocabToIndex[str];
                // Add the sample to the sequence
                wordIds.Add(index);
            }

            for (var i = 1; i < wordIds.Count - 1; i++)
            {
                cwSeqData.Add(wordIds[i]);
                nwSeqData.Add(wordIds[i + 1]);
                pwSeqData.Add(wordIds[i - 1]);
            }

            // Create input value using OneHot vector data.
            var cwValue = Value.CreateSequence<float>(vocabSize, cwSeqData, seqStartFlag, device);
            var nwValue = Value.CreateSequence<float>(vocabSize, nwSeqData, seqStartFlag, device);
            var pwValue = Value.CreateSequence<float>(vocabSize, pwSeqData, seqStartFlag, device);

            // Build input data map.
            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(cwVar, cwValue);
            inputDataMap.Add(pwVar, pwValue);
            inputDataMap.Add(nwVar, nwValue);

            // Prepare output
            const string outputName = "outputs";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Evalaute the model.
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get output result
            var outputData = new List<List<float>>();
            Value outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputData);

            //// output the result
            //var numOfElementsInSample = vocabSize;
            //uint seqNo = 0;
            //foreach (var seq in outputData)
            //{
            //    Console.Write("Seq=" + seqNo + ":");
            //    foreach (var index in seq)
            //    {
            //        // get the word based on index
            //        Console.Write(indexToVocab[index]);
            //    }
            //    Console.WriteLine();
            //    // next sequence.
            //    seqNo++;
            //}
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data as batch of sequences with variable length.
        // - how to retrieve input and output variables by name.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in the one-hot vector format.
        //
        public static void EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor device)
        {
            var vocabToIndex = buildVocabIndex(@"C:\repos\cntk\Examples\LanguageUnderstanding\ATIS\Data\ATIS.vocab");
            var indexToVocab = buildInvVocabIndex(@"C:\repos\cntk\Examples\LanguageUnderstanding\ATIS\Data\ATIS.label");

            //Function myFunc = Function.LoadModel("atis.model", device);
            Function myFunc = Function.LoadModel(@"C:\CNTKMisc\DataBackup\ATIS\work\ATIS.slot.lstm", device);

            Console.WriteLine("Evaluate batch of sequences with variable length using one-hot vector");

            // Get input variable 
            var cwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresCW")).Single();
            var nwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresNW")).Single();
            var pwVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, "featuresPW")).Single();


            uint vocabSize = cwVar.Shape.TotalSize;

            // Prepare the input data. 
            // Each sample is represented by an index to the onehot vector, so the index of the non-zero value of each sample is saved in the inner list.
            // The outer list represents sequences contained in the batch.
            var cwBatch = new List<List<uint>>();
            var pwBatch = new List<List<uint>>();
            var nwBatch = new List<List<uint>>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlagBatch = new List<bool>();

            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS i want to book a flight from new york to seattle EOS"
            };

            string[] substring;
            int numOfSequences = inputSentences.Count;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                var wordIds = new List<uint>();
                var cwSeqData = new List<uint>();
                var nwSeqData = new List<uint>();
                var pwSeqData = new List<uint>();
                
                // Get the index of each word in the sentence.
                substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    // Get the index of the word
                    var index = vocabToIndex[str];
                    // Add the sample to the sequence
                    wordIds.Add(index);
                }

                for (var i = 1; i < wordIds.Count - 1; i++)
                {
                    cwSeqData.Add(wordIds[i]);
                    nwSeqData.Add(wordIds[i + 1]);
                    pwSeqData.Add(wordIds[i - 1]);
                }
                cwBatch.Add(cwSeqData);
                nwBatch.Add(nwSeqData);
                pwBatch.Add(pwSeqData);
            }

            // Create the Value representing the batch data.
            //var cwValue = Value.CreateBatchOfSequences<float>(vocabSize, cwBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);
            //var pwValue = Value.CreateBatchOfSequences<float>(vocabSize, pwBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);
            //var nwValue = Value.CreateBatchOfSequences<float>(vocabSize, nwBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);
            var cwValue = Value.CreateBatchOfSequences<float>(vocabSize, cwBatch, DeviceDescriptor.CPUDevice);
            var pwValue = Value.CreateBatchOfSequences<float>(vocabSize, pwBatch, DeviceDescriptor.CPUDevice);
            var nwValue = Value.CreateBatchOfSequences<float>(vocabSize, nwBatch, DeviceDescriptor.CPUDevice);


            // Build input data map.
            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(cwVar, cwValue);
            inputDataMap.Add(pwVar, pwValue);
            inputDataMap.Add(nwVar, nwValue);

            // Prepare output
            const string outputName = "outputs";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Evalaute the model
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluation result.
            var outputData = new List<List<float>>();
            var outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputData);

            // output the result
            //var numOfElementsInSample = vocabSize;
            //uint seqNo = 0;
            //foreach (var seq in outputData)
            //{
            //    Console.Write("Seq=" + seqNo + ":");
            //    foreach (var index in seq)
            //    {
            //        // get the word based on index
            //        Console.Write(indexToVocab[index]);
            //    }
            //    Console.WriteLine();
            //    // next sequence.
            //    seqNo++;
            //}
        }

        private static void PrintOutput<T>(uint sampleSize, List<List<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            uint outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                uint i = 0;
                uint sampleNo = 0;
                foreach (var element in seq)
                {
                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                }
            }
        }

        private static Dictionary<string, uint> buildVocabIndex(string filePath)
        {
            var vocab = new Dictionary<string,uint>();

            string[] lines = File.ReadAllLines(filePath);
            for (uint idx = 0; idx < (uint)lines.Count(); idx++)
                vocab.Add(lines[idx], idx);

            return vocab;
        }

        private static string[] buildInvVocabIndex(string filePath)
        {
            return File.ReadAllLines(filePath);
        }
    }
}
