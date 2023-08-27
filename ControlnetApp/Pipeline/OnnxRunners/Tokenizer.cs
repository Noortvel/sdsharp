using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline.OnnxRunners
{
    //TODO: Rework on json version CLIP
    internal class Tokenizer
    {
        private readonly InferenceSession _session;

        public Tokenizer(SessionOptions options)
        {
            _session = new(ModelsPaths.Tokenizer, options);
        }

        /// <summary>
        /// Run.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>input_ids</returns>
        public IReadOnlyList<long> Run(string input)
        {
            var tensor = new DenseTensor<string>(new[] { input }, new[] { 1 });
            var namedInput = NamedOnnxValue.CreateFromTensor("string_input", tensor);

            using var result = _session.Run(new[] { namedInput });

            var inputIdsValue = result.First(); //input_ids
            var inputRawIds = inputIdsValue.AsTensor<long>().ToList();
            var idsCount = inputRawIds.Count;

            const int endOfTextToken = 49407; //<|endoftext|>
            for (int i = 0; i < 77 - idsCount; i++)
            {
                inputRawIds.Add(endOfTextToken);
            }

            return inputRawIds;
        }
    }
}
