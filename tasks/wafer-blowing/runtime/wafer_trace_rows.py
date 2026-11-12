"""Lossless JSON trace blocks with bounded decoded memory."""
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import json
import operator
import zlib


def _immutable(*args, **kwargs):
    raise TypeError('Trace rows are immutable; copy and replace the row instead.')


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON value: ' + value)


class _FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return {key: _thaw(value) for key, value in self.items()}


class _FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        return [_thaw(value) for value in self]


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


class TraceRows(Sequence):
    def __init__(self, iterable=(), *, block_size=256, cache_blocks=12, compression_level=1):
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer.')
        if type(cache_blocks) is not int or cache_blocks < 1:
            raise ValueError('cache_blocks must be a positive integer.')
        if type(compression_level) is not int or not 0 <= compression_level <= 9:
            raise ValueError('compression_level must be between 0 and 9.')
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.compression_level = compression_level
        self._blocks = []
        self._active = []
        self._active_decoded = {}
        self._cache = OrderedDict()
        self._count = 0
        self._compressed_bytes = 0
        self.extend(iterable)

    def __len__(self):
        return self._count

    def append(self, row):
        if not isinstance(row, Mapping):
            raise TypeError('A trace row must be a mapping.')
        if any(not isinstance(key, str) for key in row):
            raise TypeError('Trace row keys must be strings.')
        payload = json.dumps(dict(row), separators=(',', ':'), ensure_ascii=False,
                             allow_nan=False).encode('utf-8')
        self.append_encoded(payload, validate=False)

    def append_encoded(self, payload, *, validate=True):
        """Skip validation only for bytes from json.dumps(mapping, allow_nan=False)."""
        if not isinstance(payload, bytes):
            raise TypeError('Encoded trace rows must be UTF-8 JSON bytes.')
        if validate:
            row = json.loads(payload, parse_constant=_invalid_constant)
            if not isinstance(row, dict):
                raise TypeError('An encoded trace row must be a JSON object.')
        self._active.append(payload)
        self._count += 1
        if len(self._active) == self.block_size:
            packed = zlib.compress(b'[' + b','.join(self._active) + b']', self.compression_level)
            self._blocks.append(packed)
            self._compressed_bytes += len(packed)
            self._active = []
            self._active_decoded = {}

    def extend(self, rows):
        if rows is self:
            rows = self[:]
        for row in rows:
            self.append(row)

    def _row(self, index):
        index = operator.index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError('Trace row index out of range.')
        block, offset = divmod(index, self.block_size)
        if block == len(self._blocks):
            if offset not in self._active_decoded:
                self._active_decoded[offset] = _freeze(json.loads(self._active[offset]))
            return self._active_decoded[offset]
        if block in self._cache:
            self._cache.move_to_end(block)
        else:
            self._cache[block] = tuple(_freeze(row) for row in json.loads(zlib.decompress(self._blocks[block])))
            if len(self._cache) > self.cache_blocks:
                self._cache.popitem(last=False)
        return self._cache[block][offset]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TraceRowsView(self, range(*index.indices(self._count)))
        return self._row(index)

    def __iter__(self):
        for index in range(self._count):
            yield self._row(index)

    def storage_info(self):
        return dict(record_count=self._count, sealed_blocks=len(self._blocks),
                    compressed_bytes=self._compressed_bytes,
                    active_json_bytes=sum(map(len, self._active)),
                    active_rows=len(self._active), cached_blocks=len(self._cache),
                    cached_rows=sum(map(len, self._cache.values())) + len(self._active_decoded),
                    block_size=self.block_size, maximum_cached_blocks=self.cache_blocks)


class TraceRowsView(Sequence):
    def __init__(self, source, indices):
        self.source = source
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        selected = self.indices[index]
        return TraceRowsView(self.source, selected) if isinstance(index, slice) else self.source[selected]

    def __iter__(self):
        for index in self.indices:
            yield self.source[index]


_MISSING = object()


class _RowProxy(Mapping):
    __slots__ = ('_source', '_index', '_small_fields')

    def __init__(self, source, index):
        self._source = source
        self._index = index
        self._small_fields = None

    def __getitem__(self, key):
        if key in ('phase', 'time_s'):
            if self._small_fields is None:
                row = self._source[self._index]
                self._small_fields = (row.get('phase', _MISSING), row.get('time_s', _MISSING))
            value = self._small_fields[0 if key == 'phase' else 1]
            if value is _MISSING:
                raise KeyError(key)
            return value
        return self._source[self._index][key]

    def __iter__(self):
        return iter(self._source[self._index])

    def __len__(self):
        return len(self._source[self._index])

    def get(self, key, default=None):
        if key in ('phase', 'time_s'):
            try:
                return self[key]
            except KeyError:
                return default
        return self._source[self._index].get(key, default)

    def keys(self):
        return self._source[self._index].keys()

    def items(self):
        return self._source[self._index].items()

    def values(self):
        return self._source[self._index].values()

    def copy(self):
        return dict(self._source[self._index])

    def __deepcopy__(self, memo):
        return _thaw(self._source[self._index])


class TraceRowList(list):
    """Real list of row proxies for existing list-only protocol readers."""
    def __init__(self, iterable=(), **options):
        super().__init__()
        self.storage = TraceRows(**options)
        self.extend(iterable)

    def append(self, row):
        index = len(self.storage)
        self.storage.append(row)
        super().append(_RowProxy(self.storage, index))

    def append_encoded(self, payload, *, validate=True):
        index = len(self.storage)
        self.storage.append_encoded(payload, validate=validate)
        super().append(_RowProxy(self.storage, index))

    def extend(self, rows):
        if rows is self:
            rows = self[:]
        for row in rows:
            self.append(row)

    def __setitem__(self, index, row):
        if isinstance(index, slice):
            raise TypeError('Replace individual trace rows only.')
        normalized = operator.index(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError('Trace row index out of range.')
        stored = len(self.storage)
        self.storage.append(row)
        super().__setitem__(normalized, _RowProxy(self.storage, stored))

    def storage_info(self):
        return dict(self.storage.storage_info(), logical_rows=len(self))
