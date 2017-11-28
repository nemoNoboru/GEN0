from mutation.mutators import cross_with_mutation
from nn.models.lstm.lstm import LSTM
from pool import GeneticPool

def test_memory_lstm():
    t = LSTM(1, 2, cross_with_mutation)
    a = t.run([1])
    b = t.run([1])
    print(a)
    print(b)
    assert a[0] != b[0]

def test_diff_lstm():
    t = LSTM(1, 2, cross_with_mutation)
    q = LSTM(1, 2, cross_with_mutation)

    a = t.run([1])
    b = t.run([1])
    assert a[0] != b[0]

def test_cross_lstm():
    t = LSTM(1, 2, cross_with_mutation)
    a = t.run([1])
    t.mutateWith(t)
    b = t.run([1])
    assert a[0] == b[0]

def test_cross_lstm_another():
    t = LSTM(1, 2, cross_with_mutation)
    q = LSTM(1, 2, cross_with_mutation)
    a = t.run([1])
    t.mutateWith(q)
    b = t.run([1])
    assert a[0] != b[0]

def test_pool():
    def mock(agent):
        return 1

    t = LSTM(1, 2, cross_with_mutation)

    pool = GeneticPool(model=t, env=mock, poolSize=10)
    pool.improve()
