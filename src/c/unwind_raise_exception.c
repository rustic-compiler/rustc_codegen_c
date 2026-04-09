int _Unwind_RaiseException(void *exception_object) {
  if (__rustc_unwind_chain) {
    __rustc_unwind_chain->exception_ptr = exception_object;
    __rustc_longjmp(__rustc_unwind_chain->buf, 1);
  }
  abort();
  return 3;
}
