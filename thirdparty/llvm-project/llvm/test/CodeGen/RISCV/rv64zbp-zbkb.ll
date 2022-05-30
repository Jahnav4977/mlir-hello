; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64ZBP-ZBKB
; RUN: llc -mtriple=riscv64 -mattr=+zbkb -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=RV64ZBP-ZBKB

define signext i32 @pack_i32(i32 signext %a, i32 signext %b) nounwind {
; RV64I-LABEL: pack_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    slli a0, a0, 48
; RV64I-NEXT:    srli a0, a0, 48
; RV64I-NEXT:    slliw a1, a1, 16
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: pack_i32:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    packw a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %shl = and i32 %a, 65535
  %shl1 = shl i32 %b, 16
  %or = or i32 %shl1, %shl
  ret i32 %or
}

define i64 @pack_i64(i64 %a, i64 %b) nounwind {
; RV64I-LABEL: pack_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    slli a0, a0, 32
; RV64I-NEXT:    srli a0, a0, 32
; RV64I-NEXT:    slli a1, a1, 32
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: pack_i64:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    pack a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %shl = and i64 %a, 4294967295
  %shl1 = shl i64 %b, 32
  %or = or i64 %shl1, %shl
  ret i64 %or
}

define signext i32 @packh_i32(i32 signext %a, i32 signext %b) nounwind {
; RV64I-LABEL: packh_i32:
; RV64I:       # %bb.0:
; RV64I-NEXT:    andi a0, a0, 255
; RV64I-NEXT:    slli a1, a1, 56
; RV64I-NEXT:    srli a1, a1, 48
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: packh_i32:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    packh a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %and = and i32 %a, 255
  %and1 = shl i32 %b, 8
  %shl = and i32 %and1, 65280
  %or = or i32 %shl, %and
  ret i32 %or
}

define i32 @packh_i32_2(i32 %a, i32 %b) nounwind {
; RV64I-LABEL: packh_i32_2:
; RV64I:       # %bb.0:
; RV64I-NEXT:    andi a0, a0, 255
; RV64I-NEXT:    andi a1, a1, 255
; RV64I-NEXT:    slli a1, a1, 8
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: packh_i32_2:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    packh a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %and = and i32 %a, 255
  %and1 = and i32 %b, 255
  %shl = shl i32 %and1, 8
  %or = or i32 %shl, %and
  ret i32 %or
}

define i64 @packh_i64(i64 %a, i64 %b) nounwind {
; RV64I-LABEL: packh_i64:
; RV64I:       # %bb.0:
; RV64I-NEXT:    andi a0, a0, 255
; RV64I-NEXT:    slli a1, a1, 56
; RV64I-NEXT:    srli a1, a1, 48
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: packh_i64:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    packh a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %and = and i64 %a, 255
  %and1 = shl i64 %b, 8
  %shl = and i64 %and1, 65280
  %or = or i64 %shl, %and
  ret i64 %or
}

define i64 @packh_i64_2(i64 %a, i64 %b) nounwind {
; RV64I-LABEL: packh_i64_2:
; RV64I:       # %bb.0:
; RV64I-NEXT:    andi a0, a0, 255
; RV64I-NEXT:    andi a1, a1, 255
; RV64I-NEXT:    slli a1, a1, 8
; RV64I-NEXT:    or a0, a1, a0
; RV64I-NEXT:    ret
;
; RV64ZBP-ZBKB-LABEL: packh_i64_2:
; RV64ZBP-ZBKB:       # %bb.0:
; RV64ZBP-ZBKB-NEXT:    packh a0, a0, a1
; RV64ZBP-ZBKB-NEXT:    ret
  %and = and i64 %a, 255
  %and1 = and i64 %b, 255
  %shl = shl i64 %and1, 8
  %or = or i64 %shl, %and
  ret i64 %or
}