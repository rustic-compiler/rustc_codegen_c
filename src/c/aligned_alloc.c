#if defined(_WIN32)
void *_aligned_malloc(size_t, size_t);
void _aligned_free(void *);
#else
void free(void *);
int posix_memalign(void **, size_t, size_t);
#endif
static void *__rustc_aligned_alloc(size_t size, size_t align) {
#if defined(_WIN32)
  return _aligned_malloc(size, align);
#else
  if (align < sizeof(void *)) align = sizeof(void *);
  void *ptr;
  if (posix_memalign(&ptr, align, size) != 0) return (void *)0;
  return ptr;
#endif
}
static void __rustc_aligned_free(void *ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
