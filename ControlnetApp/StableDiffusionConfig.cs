namespace ControlnetApp
{
    internal class StableDiffusionConfig
    {
        public int Height { get; init; } = 512;

        public int Width { get; init; } = 512;

        public int NumInferenceSteps { get; init; } = 15;

        public float GuidanceScale { get; init; } = 7.5f;

        public float ControlnetConditionScale { get; init; } = 1.0f;

        public float ControlnetUnConditionScale { get; init; } = 1.0f;
    }
}
