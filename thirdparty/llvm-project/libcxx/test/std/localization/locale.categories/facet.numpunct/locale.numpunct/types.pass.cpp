//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT>
// class numpunct
//     : public locale::facet
// {
// public:
//     typedef charT char_type;
//     typedef basic_string<charT> string_type;
//     static locale::id id;

#include <locale>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        assert(std::has_facet<std::numpunct<char> >(l));
        const std::numpunct<char>& f = std::use_facet<std::numpunct<char> >(l);
        ((void)f); // Prevent unused warning
        {
            (void)std::numpunct<char>::id;
        }
        static_assert((std::is_same<std::numpunct<char>::char_type, char>::value), "");
        static_assert((std::is_same<std::numpunct<char>::string_type, std::string>::value), "");
        static_assert((std::is_base_of<std::locale::facet, std::numpunct<char> >::value), "");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        assert(std::has_facet<std::numpunct<wchar_t> >(l));
        const std::numpunct<wchar_t>& f = std::use_facet<std::numpunct<wchar_t> >(l);
        ((void)f); // Prevent unused warning
        {
            (void)std::numpunct<wchar_t>::id;
        }
        static_assert((std::is_same<std::numpunct<wchar_t>::char_type, wchar_t>::value), "");
        static_assert((std::is_same<std::numpunct<wchar_t>::string_type, std::wstring>::value), "");
        static_assert((std::is_base_of<std::locale::facet, std::numpunct<wchar_t> >::value), "");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}