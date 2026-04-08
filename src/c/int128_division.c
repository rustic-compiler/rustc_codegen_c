
typedef unsigned __int128 __rustc_u128;
typedef __int128 __rustc_i128;

__attribute__((weak)) __rustc_u128 __udivti3(__rustc_u128 n, __rustc_u128 d) {
  if ((uint64_t)(n >> 64) == 0 && (uint64_t)(d >> 64) == 0)
    return (uint64_t)n / (uint64_t)d;
  __rustc_u128 q = 0, r = 0;
  for (int i = 127; i >= 0; --i) {
    r = (r << 1) | ((n >> i) & 1);
    if (r >= d) { r -= d; q |= (__rustc_u128)1 << i; }
  }
  return q;
}

__attribute__((weak)) __rustc_u128 __umodti3(__rustc_u128 n, __rustc_u128 d) {
  if ((uint64_t)(n >> 64) == 0 && (uint64_t)(d >> 64) == 0)
    return (uint64_t)n % (uint64_t)d;
  __rustc_u128 r = 0;
  for (int i = 127; i >= 0; --i) {
    r = (r << 1) | ((n >> i) & 1);
    if (r >= d) r -= d;
  }
  return r;
}

__attribute__((weak)) __rustc_i128 __divti3(__rustc_i128 n, __rustc_i128 d) {
  int neg = (n < 0) != (d < 0);
  __rustc_u128 un = n < 0 ? -(__rustc_u128)n : (__rustc_u128)n;
  __rustc_u128 ud = d < 0 ? -(__rustc_u128)d : (__rustc_u128)d;
  __rustc_u128 uq = __udivti3(un, ud);
  return neg ? -(__rustc_i128)uq : (__rustc_i128)uq;
}

__attribute__((weak)) __rustc_i128 __modti3(__rustc_i128 n, __rustc_i128 d) {
  int neg = n < 0;
  __rustc_u128 un = n < 0 ? -(__rustc_u128)n : (__rustc_u128)n;
  __rustc_u128 ud = d < 0 ? -(__rustc_u128)d : (__rustc_u128)d;
  __rustc_u128 ur = __umodti3(un, ud);
  return neg ? -(__rustc_i128)ur : (__rustc_i128)ur;
}
