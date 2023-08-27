namespace ControlnetApp.Domain;

public record MocapResult(
    Size Info,
    IReadOnlyList<IReadOnlyList<Vector2i>> Points
);
