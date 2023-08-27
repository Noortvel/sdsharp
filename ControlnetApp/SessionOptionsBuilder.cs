using ControlnetApp.Domain;
using Microsoft.ML.OnnxRuntime;

namespace ControlnetApp
{
    internal class SessionOptionsBuilder
    {
        public static SessionOptions Build(ExecutionProvider executionProvider)
        {
            var deviceId = 1;

            SessionOptions sessionOptions;
            switch (executionProvider)
            {
                case ExecutionProvider.Cpu:
                    sessionOptions = new SessionOptions();
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;

                case ExecutionProvider.Cuda:
                    sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId);
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    break;

                case ExecutionProvider.DML: //WORK ONLY IN x64
                    sessionOptions = new SessionOptions();
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.AppendExecutionProvider_DML(deviceId);
                    break;

                default:
                    throw new ArgumentException("Unknown ExecutionProvider");
            }

            sessionOptions.RegisterCustomOpLibraryV2("ortextensions.dll", out var libraryHandle);

            return sessionOptions;
        }
    }
}
