int __rust_try(void (*try_fn)(void *), void *data, void (*catch_fn)(void *, void *)) {
  struct __rustc_unwind_context __ctx;
  __ctx.prev = __rustc_unwind_chain;
  __ctx.exception_ptr = (void *)0;
  __rustc_unwind_chain = &__ctx;
  if (__rustc_setjmp(__ctx.buf) == 0) {
    try_fn(data);
    __rustc_unwind_chain = __ctx.prev;
    return 0;
  } else {
    void *__exn = __ctx.exception_ptr;
    __rustc_unwind_chain = __ctx.prev;
    catch_fn(data, __exn);
    return 1;
  }
}
