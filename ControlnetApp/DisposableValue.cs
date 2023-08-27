namespace ControlnetApp
{
    internal struct DisposableValue<TValue> : IDisposable
        where TValue : class
    {
        private readonly IDisposable _disposable;

        public DisposableValue(
            TValue value,
            IDisposable disposable) 
        {
            _disposable = disposable;
            Value = value;
        }

        public TValue Value { get; }

        public void Dispose()
        {
            _disposable.Dispose();
        }
    }
}
