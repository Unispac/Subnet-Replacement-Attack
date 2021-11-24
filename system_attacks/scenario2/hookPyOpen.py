def open(*args,**kw):
  print(*args, **kw)
  ret = _open(*args,**kw)
  return ret

_open = __builtins__.open
__builtins__.open = open
__builtins__.file = open
