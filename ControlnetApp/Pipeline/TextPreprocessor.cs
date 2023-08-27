using ControlnetApp.Pipeline.OnnxRunners;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline
{
    internal class TextPreprocessor
    {
        private readonly Tokenizer _tokenizer;
        private readonly TextEncoder _textEncoder;

        public TextPreprocessor(Tokenizer tokenizer, TextEncoder textEncoder)
        {
            _tokenizer = tokenizer;
            _textEncoder = textEncoder;
        }


        public DenseTensor<float> Run(
            string positivePrompt,
            bool isGuadiance,
            string? negativePrompt = default)
        {
            var batch = isGuadiance ? 2 : 1;
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { batch, 77, 768 });

            var tokens = _tokenizer.Run(positivePrompt);
            using var encodedText = _textEncoder.Run(tokens.Select(x => (int)x).ToArray());

            if (isGuadiance)
            {
                int[] negativeTokens;
                if (negativePrompt != default)
                {
                    negativeTokens = _tokenizer.Run(negativePrompt).Select(x => (int)x).ToArray();
                }
                else
                {
                    negativeTokens = CreateUncondInput();
                }

                using var negativeEncodedText = _textEncoder.Run(negativeTokens);

                for (int i = 0; i < 77; i++)
                {
                    for (int j = 0; j < 768; j++)
                    {
                        textEmbeddings[0, i, j] = negativeEncodedText.Value[0, i, j];
                        textEmbeddings[1, i, j] = encodedText.Value[0, i, j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < 77; i++)
                {
                    for (int j = 0; j < 768; j++)
                    {
                        textEmbeddings[0, i, j] = encodedText.Value[0, i, j];
                    }
                }
            }

            return textEmbeddings;
        }

        private static int[] CreateUncondInput()
        {
            // Create an array of empty tokens for the unconditional input.
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            var inputIds = new List<Int32>();
            inputIds.Add(49406);
            var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            return inputIds.ToArray();
        }
    }
}
