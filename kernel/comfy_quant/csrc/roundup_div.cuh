/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef COMFY_ROUNDUP_DIV_CUH_
#define COMFY_ROUNDUP_DIV_CUH_

#include <type_traits>

namespace comfy {

template <typename T>
constexpr T RoundUpDivide(const T x, const T y) {
  static_assert(std::is_integral<T>::value, "RoundUpDivide requires integral types.");
  static_assert(std::is_unsigned<T>::value, "RoundUpDivide requires unsigned types.");
  return (x + (y - 1)) / y;
}

} // namespace comfy

#endif // COMFY_ROUNDUP_DIV_CUH_
