; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py
; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1010 < %s | FileCheck -check-prefixes=ALL,FAST %s
; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=ALL,FAST %s
; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,FAST %s
; RUN: opt -passes='print<cost-model>' 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL,SLOW %s

; RUN: opt -passes='print<cost-model>' -cost-kind=code-size 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1010 < %s | FileCheck -check-prefixes=ALL-SIZE,FAST-SIZE %s
; RUN: opt -passes='print<cost-model>' -cost-kind=code-size 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=ALL-SIZE,FAST-SIZE %s
; RUN: opt -passes='print<cost-model>' -cost-kind=code-size 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL-SIZE,FAST-SIZE %s
; RUN: opt -passes='print<cost-model>' -cost-kind=code-size 2>&1 -disable-output -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL-SIZE,SLOW-SIZE %s
; END.

define i32 @add(i32 %arg) {
  ; -- Same size registeres --
; ALL-LABEL: 'add'
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A = zext <4 x i1> undef to <4 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B = sext <4 x i1> undef to <4 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = trunc <4 x i32> undef to <4 x i1>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D = zext <8 x i1> undef to <8 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %E = sext <8 x i1> undef to <8 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %F = trunc <8 x i32> undef to <8 x i1>
; ALL-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %G = zext i1 undef to i32
; ALL-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %H = trunc i32 undef to i1
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret i32 undef
;
; ALL-SIZE-LABEL: 'add'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A = zext <4 x i1> undef to <4 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B = sext <4 x i1> undef to <4 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = trunc <4 x i32> undef to <4 x i1>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D = zext <8 x i1> undef to <8 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %E = sext <8 x i1> undef to <8 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %F = trunc <8 x i32> undef to <8 x i1>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %G = zext i1 undef to i32
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %H = trunc i32 undef to i1
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret i32 undef
;
  %A = zext <4 x i1> undef to <4 x i32>
  %B = sext <4 x i1> undef to <4 x i32>
  %C = trunc <4 x i32> undef to <4 x i1>

  ; -- Different size registers --
  %D = zext <8 x i1> undef to <8 x i32>
  %E = sext <8 x i1> undef to <8 x i32>
  %F = trunc <8 x i32> undef to <8 x i1>

  ; -- scalars --
  %G = zext i1 undef to i32
  %H = trunc i32 undef to i1

  ret i32 undef
}

define i32 @zext_sext(<8 x i1> %in) {
; FAST-LABEL: 'zext_sext'
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A1 = zext <16 x i8> undef to <16 x i16>
; FAST-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A2 = sext <16 x i8> undef to <16 x i16>
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A = sext <8 x i16> undef to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %B = zext <8 x i16> undef to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = sext <4 x i32> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i16.s = sext <4 x i16> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i8.s = sext <4 x i8> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D = zext <4 x i32> undef to <4 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D1 = zext <8 x i32> undef to <8 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D2 = sext <8 x i32> undef to <8 x i64>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D3 = zext <16 x i16> undef to <16 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D4 = zext <16 x i8> undef to <16 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D5 = zext <16 x i1> undef to <16 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %E = trunc <4 x i64> undef to <4 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F = trunc <8 x i32> undef to <8 x i16>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F1 = trunc <16 x i16> undef to <16 x i8>
; FAST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %F2 = trunc <8 x i32> undef to <8 x i8>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %F3 = trunc <4 x i64> undef to <4 x i8>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G = trunc <8 x i64> undef to <8 x i32>
; FAST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G1 = trunc <16 x i32> undef to <16 x i16>
; FAST-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %G2 = trunc <16 x i32> undef to <16 x i8>
; FAST-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret i32 undef
;
; SLOW-LABEL: 'zext_sext'
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A1 = zext <16 x i8> undef to <16 x i16>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A2 = sext <16 x i8> undef to <16 x i16>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A = sext <8 x i16> undef to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B = zext <8 x i16> undef to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = sext <4 x i32> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i16.s = sext <4 x i16> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i8.s = sext <4 x i8> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D = zext <4 x i32> undef to <4 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D1 = zext <8 x i32> undef to <8 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D2 = sext <8 x i32> undef to <8 x i64>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D3 = zext <16 x i16> undef to <16 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D4 = zext <16 x i8> undef to <16 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D5 = zext <16 x i1> undef to <16 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %E = trunc <4 x i64> undef to <4 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F = trunc <8 x i32> undef to <8 x i16>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F1 = trunc <16 x i16> undef to <16 x i8>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F2 = trunc <8 x i32> undef to <8 x i8>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F3 = trunc <4 x i64> undef to <4 x i8>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G = trunc <8 x i64> undef to <8 x i32>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G1 = trunc <16 x i32> undef to <16 x i16>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G2 = trunc <16 x i32> undef to <16 x i8>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret i32 undef
;
; FAST-SIZE-LABEL: 'zext_sext'
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A1 = zext <16 x i8> undef to <16 x i16>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A2 = sext <16 x i8> undef to <16 x i16>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A = sext <8 x i16> undef to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %B = zext <8 x i16> undef to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = sext <4 x i32> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i16.s = sext <4 x i16> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i8.s = sext <4 x i8> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D = zext <4 x i32> undef to <4 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D1 = zext <8 x i32> undef to <8 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D2 = sext <8 x i32> undef to <8 x i64>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D3 = zext <16 x i16> undef to <16 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D4 = zext <16 x i8> undef to <16 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D5 = zext <16 x i1> undef to <16 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %E = trunc <4 x i64> undef to <4 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F = trunc <8 x i32> undef to <8 x i16>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F1 = trunc <16 x i16> undef to <16 x i8>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %F2 = trunc <8 x i32> undef to <8 x i8>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %F3 = trunc <4 x i64> undef to <4 x i8>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G = trunc <8 x i64> undef to <8 x i32>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G1 = trunc <16 x i32> undef to <16 x i16>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %G2 = trunc <16 x i32> undef to <16 x i8>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret i32 undef
;
; SLOW-SIZE-LABEL: 'zext_sext'
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A1 = zext <16 x i8> undef to <16 x i16>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %A2 = sext <16 x i8> undef to <16 x i16>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A = sext <8 x i16> undef to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B = zext <8 x i16> undef to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C = sext <4 x i32> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i16.s = sext <4 x i16> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C.v4i8.s = sext <4 x i8> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D = zext <4 x i32> undef to <4 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %D1 = zext <8 x i32> undef to <8 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D2 = sext <8 x i32> undef to <8 x i64>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D3 = zext <16 x i16> undef to <16 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D4 = zext <16 x i8> undef to <16 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 16 for instruction: %D5 = zext <16 x i1> undef to <16 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %E = trunc <4 x i64> undef to <4 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F = trunc <8 x i32> undef to <8 x i16>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F1 = trunc <16 x i16> undef to <16 x i8>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F2 = trunc <8 x i32> undef to <8 x i8>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %F3 = trunc <4 x i64> undef to <4 x i8>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G = trunc <8 x i64> undef to <8 x i32>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G1 = trunc <16 x i32> undef to <16 x i16>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %G2 = trunc <16 x i32> undef to <16 x i8>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret i32 undef
;
  %Z = zext <8 x i1> %in to <8 x i32>
  %S = sext <8 x i1> %in to <8 x i32>

  %A1 = zext <16 x i8> undef to <16 x i16>
  %A2 = sext <16 x i8> undef to <16 x i16>
  %A = sext <8 x i16> undef to <8 x i32>
  %B = zext <8 x i16> undef to <8 x i32>
  %C = sext <4 x i32> undef to <4 x i64>

  %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
  %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
  %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
  %C.v4i16.s = sext <4 x i16> undef to <4 x i64>

  %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
  %C.v4i8.s = sext <4 x i8> undef to <4 x i64>

  %D = zext <4 x i32> undef to <4 x i64>

  %D1 = zext <8 x i32> undef to <8 x i64>

  %D2 = sext <8 x i32> undef to <8 x i64>

  %D3 = zext <16 x i16> undef to <16 x i32>
  %D4 = zext <16 x i8> undef to <16 x i32>
  %D5 = zext <16 x i1> undef to <16 x i32>

  %E = trunc <4 x i64> undef to <4 x i32>
  %F = trunc <8 x i32> undef to <8 x i16>
  %F1 = trunc <16 x i16> undef to <16 x i8>
  %F2 = trunc <8 x i32> undef to <8 x i8>
  %F3 = trunc <4 x i64> undef to <4 x i8>

  %G = trunc <8 x i64> undef to <8 x i32>
  %G1 = trunc <16 x i32> undef to <16 x i16>
  %G2 = trunc <16 x i32> undef to <16 x i8>
  ret i32 undef
}

define i32 @masks8(<8 x i1> %in) {
; ALL-LABEL: 'masks8'
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret i32 undef
;
; ALL-SIZE-LABEL: 'masks8'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %Z = zext <8 x i1> %in to <8 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %S = sext <8 x i1> %in to <8 x i32>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret i32 undef
;
  %Z = zext <8 x i1> %in to <8 x i32>
  %S = sext <8 x i1> %in to <8 x i32>
  ret i32 undef
}

define i32 @masks4(<4 x i1> %in) {
; ALL-LABEL: 'masks4'
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %Z = zext <4 x i1> %in to <4 x i64>
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %S = sext <4 x i1> %in to <4 x i64>
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret i32 undef
;
; ALL-SIZE-LABEL: 'masks4'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %Z = zext <4 x i1> %in to <4 x i64>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %S = sext <4 x i1> %in to <4 x i64>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret i32 undef
;
  %Z = zext <4 x i1> %in to <4 x i64>
  %S = sext <4 x i1> %in to <4 x i64>
  ret i32 undef
}

define void @sitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
; FAST-LABEL: 'sitofp4'
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = sitofp <4 x i1> %a to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = sitofp <4 x i1> %a to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = sitofp <4 x i8> %b to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = sitofp <4 x i8> %b to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %C1 = sitofp <4 x i16> %c to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = sitofp <4 x i16> %c to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = sitofp <4 x i32> %d to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = sitofp <4 x i32> %d to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; SLOW-LABEL: 'sitofp4'
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = sitofp <4 x i1> %a to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = sitofp <4 x i1> %a to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = sitofp <4 x i8> %b to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = sitofp <4 x i8> %b to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C1 = sitofp <4 x i16> %c to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = sitofp <4 x i16> %c to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = sitofp <4 x i32> %d to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = sitofp <4 x i32> %d to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; FAST-SIZE-LABEL: 'sitofp4'
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = sitofp <4 x i1> %a to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = sitofp <4 x i1> %a to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = sitofp <4 x i8> %b to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = sitofp <4 x i8> %b to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %C1 = sitofp <4 x i16> %c to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = sitofp <4 x i16> %c to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = sitofp <4 x i32> %d to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = sitofp <4 x i32> %d to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
; SLOW-SIZE-LABEL: 'sitofp4'
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = sitofp <4 x i1> %a to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = sitofp <4 x i1> %a to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = sitofp <4 x i8> %b to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = sitofp <4 x i8> %b to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C1 = sitofp <4 x i16> %c to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = sitofp <4 x i16> %c to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = sitofp <4 x i32> %d to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = sitofp <4 x i32> %d to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
  %A1 = sitofp <4 x i1> %a to <4 x float>
  %A2 = sitofp <4 x i1> %a to <4 x double>
  %B1 = sitofp <4 x i8> %b to <4 x float>
  %B2 = sitofp <4 x i8> %b to <4 x double>
  %C1 = sitofp <4 x i16> %c to <4 x float>
  %C2 = sitofp <4 x i16> %c to <4 x double>
  %D1 = sitofp <4 x i32> %d to <4 x float>
  %D2 = sitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @sitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
; ALL-LABEL: 'sitofp8'
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A1 = sitofp <8 x i1> %a to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B1 = sitofp <8 x i8> %b to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C1 = sitofp <8 x i16> %c to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D1 = sitofp <8 x i32> %d to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; ALL-SIZE-LABEL: 'sitofp8'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A1 = sitofp <8 x i1> %a to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B1 = sitofp <8 x i8> %b to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C1 = sitofp <8 x i16> %c to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D1 = sitofp <8 x i32> %d to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
  %A1 = sitofp <8 x i1> %a to <8 x float>
  %B1 = sitofp <8 x i8> %b to <8 x float>
  %C1 = sitofp <8 x i16> %c to <8 x float>
  %D1 = sitofp <8 x i32> %d to <8 x float>
  ret void
}

define void @uitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
; FAST-LABEL: 'uitofp4'
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = uitofp <4 x i1> %a to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = uitofp <4 x i1> %a to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = uitofp <4 x i8> %b to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = uitofp <4 x i8> %b to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %C1 = uitofp <4 x i16> %c to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = uitofp <4 x i16> %c to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = uitofp <4 x i32> %d to <4 x float>
; FAST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = uitofp <4 x i32> %d to <4 x double>
; FAST-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; SLOW-LABEL: 'uitofp4'
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = uitofp <4 x i1> %a to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = uitofp <4 x i1> %a to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = uitofp <4 x i8> %b to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = uitofp <4 x i8> %b to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C1 = uitofp <4 x i16> %c to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = uitofp <4 x i16> %c to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = uitofp <4 x i32> %d to <4 x float>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = uitofp <4 x i32> %d to <4 x double>
; SLOW-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; FAST-SIZE-LABEL: 'uitofp4'
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = uitofp <4 x i1> %a to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = uitofp <4 x i1> %a to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = uitofp <4 x i8> %b to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = uitofp <4 x i8> %b to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %C1 = uitofp <4 x i16> %c to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = uitofp <4 x i16> %c to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = uitofp <4 x i32> %d to <4 x float>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = uitofp <4 x i32> %d to <4 x double>
; FAST-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
; SLOW-SIZE-LABEL: 'uitofp4'
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = uitofp <4 x i1> %a to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A2 = uitofp <4 x i1> %a to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B1 = uitofp <4 x i8> %b to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %B2 = uitofp <4 x i8> %b to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C1 = uitofp <4 x i16> %c to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %C2 = uitofp <4 x i16> %c to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %D1 = uitofp <4 x i32> %d to <4 x float>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %D2 = uitofp <4 x i32> %d to <4 x double>
; SLOW-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
  %A1 = uitofp <4 x i1> %a to <4 x float>
  %A2 = uitofp <4 x i1> %a to <4 x double>
  %B1 = uitofp <4 x i8> %b to <4 x float>
  %B2 = uitofp <4 x i8> %b to <4 x double>
  %C1 = uitofp <4 x i16> %c to <4 x float>
  %C2 = uitofp <4 x i16> %c to <4 x double>
  %D1 = uitofp <4 x i32> %d to <4 x float>
  %D2 = uitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @uitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
; ALL-LABEL: 'uitofp8'
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A1 = uitofp <8 x i1> %a to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B1 = uitofp <8 x i8> %b to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C1 = uitofp <8 x i16> %c to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D1 = uitofp <8 x i32> %d to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; ALL-SIZE-LABEL: 'uitofp8'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A1 = uitofp <8 x i1> %a to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %B1 = uitofp <8 x i8> %b to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %C1 = uitofp <8 x i16> %c to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %D1 = uitofp <8 x i32> %d to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
  %A1 = uitofp <8 x i1> %a to <8 x float>
  %B1 = uitofp <8 x i8> %b to <8 x float>
  %C1 = uitofp <8 x i16> %c to <8 x float>
  %D1 = uitofp <8 x i32> %d to <8 x float>
  ret void
}

define void @fp_conv(<8 x float> %a, <16 x float>%b, <4 x float> %c) {
; ALL-LABEL: 'fp_conv'
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = fpext <4 x float> %c to <4 x double>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A2 = fpext <8 x float> %a to <8 x double>
; ALL-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A3 = fptrunc <4 x double> undef to <4 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A4 = fptrunc <8 x double> undef to <8 x float>
; ALL-NEXT:  Cost Model: Found an estimated cost of 10 for instruction: ret void
;
; ALL-SIZE-LABEL: 'fp_conv'
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A1 = fpext <4 x float> %c to <4 x double>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A2 = fpext <8 x float> %a to <8 x double>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %A3 = fptrunc <4 x double> undef to <4 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %A4 = fptrunc <8 x double> undef to <8 x float>
; ALL-SIZE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: ret void
;
  %A1 = fpext <4 x float> %c to <4 x double>
  %A2 = fpext <8 x float> %a to <8 x double>
  %A3 = fptrunc <4 x double> undef to <4 x float>
  %A4 = fptrunc <8 x double> undef to <8 x float>
  ret void
}