; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -o - -mtriple=aarch64-none-linux-gnu -code-model=tiny < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -o - -mtriple=aarch64-none-linux-gnu -code-model=tiny -fast-isel < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -o - -mtriple=aarch64-none-linux-gnu -code-model=tiny -global-isel < %s | FileCheck %s --check-prefix=CHECK-GLOBISEL

; Note fast-isel tests here will fall back to isel

@src = external local_unnamed_addr global [65536 x i8], align 1
@dst = external global [65536 x i8], align 1
@ptr = external local_unnamed_addr global i8*, align 8

define dso_local void @foo1() {
; CHECK-LABEL: foo1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x8, :got:src
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    ldr x9, :got:dst
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: foo1:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    ldr x8, :got:src
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    ldr x9, :got:dst
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @src, i64 0, i64 0), align 1
  store i8 %0, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @dst, i64 0, i64 0), align 1
  ret void
}

define dso_local void @foo2() {
; CHECK-LABEL: foo2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x8, :got:ptr
; CHECK-NEXT:    ldr x9, :got:dst
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: foo2:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    ldr x8, :got:ptr
; CHECK-GLOBISEL-NEXT:    ldr x9, :got:dst
; CHECK-GLOBISEL-NEXT:    str x9, [x8]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  store i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @dst, i64 0, i64 0), i8** @ptr, align 8
  ret void
}

define dso_local void @foo3() {
; FIXME: Needn't adr ptr
;
; CHECK-LABEL: foo3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x8, :got:src
; CHECK-NEXT:    ldr x9, :got:ptr
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    ldr x9, [x9]
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: foo3:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    ldr x8, :got:src
; CHECK-GLOBISEL-NEXT:    ldr x9, :got:ptr
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    ldr x9, [x9]
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @src, i64 0, i64 0), align 1
  %1 = load i8*, i8** @ptr, align 8
  store i8 %0, i8* %1, align 1
  ret void
}

@lsrc = internal global i8 0, align 4
@ldst = internal global i8 0, align 4
@lptr = internal global i8* null, align 8

define dso_local void @bar1() {
; CHECK-LABEL: bar1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lsrc
; CHECK-NEXT:    adr x9, ldst
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: bar1:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lsrc
; CHECK-GLOBISEL-NEXT:    adr x9, ldst
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* @lsrc, align 4
  store i8 %0, i8* @ldst, align 4
  ret void
}

define dso_local void @bar2() {
; CHECK-LABEL: bar2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lptr
; CHECK-NEXT:    adr x9, ldst
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: bar2:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lptr
; CHECK-GLOBISEL-NEXT:    adr x9, ldst
; CHECK-GLOBISEL-NEXT:    str x9, [x8]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  store i8* @ldst, i8** @lptr, align 8
  ret void
}

define dso_local void @bar3() {
; FIXME: Needn't adr lptr
;
; CHECK-LABEL: bar3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lsrc
; CHECK-NEXT:    ldr x9, lptr
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: bar3:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lsrc
; CHECK-GLOBISEL-NEXT:    adr x9, lptr
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    ldr x9, [x9]
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* @lsrc, align 4
  %1 = load i8*, i8** @lptr, align 8
  store i8 %0, i8* %1, align 1
  ret void
}


@lbsrc = internal global [65536 x i8] zeroinitializer, align 4
@lbdst = internal global [65536 x i8] zeroinitializer, align 4

define dso_local void @baz1() {
; CHECK-LABEL: baz1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lbsrc
; CHECK-NEXT:    adr x9, lbdst
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: baz1:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lbsrc
; CHECK-GLOBISEL-NEXT:    adr x9, lbdst
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @lbsrc, i64 0, i64 0), align 4
  store i8 %0, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @lbdst, i64 0, i64 0), align 4
  ret void
}

define dso_local void @baz2() {
; CHECK-LABEL: baz2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lptr
; CHECK-NEXT:    adr x9, lbdst
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: baz2:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lptr
; CHECK-GLOBISEL-NEXT:    adr x9, lbdst
; CHECK-GLOBISEL-NEXT:    str x9, [x8]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  store i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @lbdst, i64 0, i64 0), i8** @lptr, align 8
  ret void
}

define dso_local void @baz3() {
; FIXME: Needn't adr lptr
;
; CHECK-LABEL: baz3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x8, lbsrc
; CHECK-NEXT:    ldr x9, lptr
; CHECK-NEXT:    ldrb w8, [x8]
; CHECK-NEXT:    strb w8, [x9]
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: baz3:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x8, lbsrc
; CHECK-GLOBISEL-NEXT:    adr x9, lptr
; CHECK-GLOBISEL-NEXT:    ldrb w8, [x8]
; CHECK-GLOBISEL-NEXT:    ldr x9, [x9]
; CHECK-GLOBISEL-NEXT:    strb w8, [x9]
; CHECK-GLOBISEL-NEXT:    ret
entry:
  %0 = load i8, i8* getelementptr inbounds ([65536 x i8], [65536 x i8]* @lbsrc, i64 0, i64 0), align 4
  %1 = load i8*, i8** @lptr, align 8
  store i8 %0, i8* %1, align 1
  ret void
}


declare void @func(...)

define dso_local i8* @externfuncaddr() {
; CHECK-LABEL: externfuncaddr:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x0, :got:func
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: externfuncaddr:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    ldr x0, :got:func
; CHECK-GLOBISEL-NEXT:    ret
entry:
      ret i8* bitcast (void (...)* @func to i8*)
}

define dso_local i8* @localfuncaddr() {
; CHECK-LABEL: localfuncaddr:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr x0, externfuncaddr
; CHECK-NEXT:    ret
;
; CHECK-GLOBISEL-LABEL: localfuncaddr:
; CHECK-GLOBISEL:       // %bb.0: // %entry
; CHECK-GLOBISEL-NEXT:    adr x0, externfuncaddr
; CHECK-GLOBISEL-NEXT:    ret
entry:
      ret i8* bitcast (i8* ()* @externfuncaddr to i8*)
}