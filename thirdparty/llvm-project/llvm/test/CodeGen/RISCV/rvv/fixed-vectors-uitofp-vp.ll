; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+m,+v,+zfh,+experimental-zvfh \
; RUN:     -riscv-v-vector-bits-min=128 < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zfh,+experimental-zvfh \
; RUN:     -riscv-v-vector-bits-min=128 < %s | FileCheck %s

declare <4 x half> @llvm.vp.uitofp.v4f16.v4i7(<4 x i7>, <4 x i1>, i32)

define <4 x half> @vuitofp_v4f16_v4i7(<4 x i7> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a1, 127
; CHECK-NEXT:    vsetivli zero, 4, e8, mf4, ta, mu
; CHECK-NEXT:    vand.vx v9, v8, a1
; CHECK-NEXT:    vsetvli zero, a0, e8, mf4, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v9, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i7(<4 x i7> %va, <4 x i1> %m, i32 %evl)
  ret <4 x half> %v
}

declare <4 x half> @llvm.vp.uitofp.v4f16.v4i8(<4 x i8>, <4 x i1>, i32)

define <4 x half> @vuitofp_v4f16_v4i8(<4 x i8> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e8, mf4, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i8(<4 x i8> %va, <4 x i1> %m, i32 %evl)
  ret <4 x half> %v
}

define <4 x half> @vuitofp_v4f16_v4i8_unmasked(<4 x i8> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i8_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e8, mf4, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i8(<4 x i8> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x half> %v
}

declare <4 x half> @llvm.vp.uitofp.v4f16.v4i16(<4 x i16>, <4 x i1>, i32)

define <4 x half> @vuitofp_v4f16_v4i16(<4 x i16> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i16(<4 x i16> %va, <4 x i1> %m, i32 %evl)
  ret <4 x half> %v
}

define <4 x half> @vuitofp_v4f16_v4i16_unmasked(<4 x i16> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i16_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i16(<4 x i16> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x half> %v
}

declare <4 x half> @llvm.vp.uitofp.v4f16.v4i32(<4 x i32>, <4 x i1>, i32)

define <4 x half> @vuitofp_v4f16_v4i32(<4 x i32> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i32(<4 x i32> %va, <4 x i1> %m, i32 %evl)
  ret <4 x half> %v
}

define <4 x half> @vuitofp_v4f16_v4i32_unmasked(<4 x i32> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i32(<4 x i32> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x half> %v
}

declare <4 x half> @llvm.vp.uitofp.v4f16.v4i64(<4 x i64>, <4 x i1>, i32)

define <4 x half> @vuitofp_v4f16_v4i64(<4 x i64> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v10, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, mu
; CHECK-NEXT:    vfncvt.f.f.w v8, v10, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i64(<4 x i64> %va, <4 x i1> %m, i32 %evl)
  ret <4 x half> %v
}

define <4 x half> @vuitofp_v4f16_v4i64_unmasked(<4 x i64> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f16_v4i64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v10, v8
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, mu
; CHECK-NEXT:    vfncvt.f.f.w v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x half> @llvm.vp.uitofp.v4f16.v4i64(<4 x i64> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x half> %v
}

declare <4 x float> @llvm.vp.uitofp.v4f32.v4i8(<4 x i8>, <4 x i1>, i32)

define <4 x float> @vuitofp_v4f32_v4i8(<4 x i8> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vzext.vf2 v9, v8, v0.t
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v9, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i8(<4 x i8> %va, <4 x i1> %m, i32 %evl)
  ret <4 x float> %v
}

define <4 x float> @vuitofp_v4f32_v4i8_unmasked(<4 x i8> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i8_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vzext.vf2 v9, v8
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i8(<4 x i8> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x float> %v
}

declare <4 x float> @llvm.vp.uitofp.v4f32.v4i16(<4 x i16>, <4 x i1>, i32)

define <4 x float> @vuitofp_v4f32_v4i16(<4 x i16> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i16(<4 x i16> %va, <4 x i1> %m, i32 %evl)
  ret <4 x float> %v
}

define <4 x float> @vuitofp_v4f32_v4i16_unmasked(<4 x i16> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i16_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i16(<4 x i16> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x float> %v
}

declare <4 x float> @llvm.vp.uitofp.v4f32.v4i32(<4 x i32>, <4 x i1>, i32)

define <4 x float> @vuitofp_v4f32_v4i32(<4 x i32> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i32(<4 x i32> %va, <4 x i1> %m, i32 %evl)
  ret <4 x float> %v
}

define <4 x float> @vuitofp_v4f32_v4i32_unmasked(<4 x i32> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i32(<4 x i32> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x float> %v
}

declare <4 x float> @llvm.vp.uitofp.v4f32.v4i64(<4 x i64>, <4 x i1>, i32)

define <4 x float> @vuitofp_v4f32_v4i64(<4 x i64> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v10, v8, v0.t
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i64(<4 x i64> %va, <4 x i1> %m, i32 %evl)
  ret <4 x float> %v
}

define <4 x float> @vuitofp_v4f32_v4i64_unmasked(<4 x i64> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f32_v4i64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v10, v8
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x float> @llvm.vp.uitofp.v4f32.v4i64(<4 x i64> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x float> %v
}

declare <4 x double> @llvm.vp.uitofp.v4f64.v4i8(<4 x i8>, <4 x i1>, i32)

define <4 x double> @vuitofp_v4f64_v4i8(<4 x i8> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vzext.vf4 v10, v8, v0.t
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v10, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i8(<4 x i8> %va, <4 x i1> %m, i32 %evl)
  ret <4 x double> %v
}

define <4 x double> @vuitofp_v4f64_v4i8_unmasked(<4 x i8> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i8_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vzext.vf4 v10, v8
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i8(<4 x i8> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x double> %v
}

declare <4 x double> @llvm.vp.uitofp.v4f64.v4i16(<4 x i16>, <4 x i1>, i32)

define <4 x double> @vuitofp_v4f64_v4i16(<4 x i16> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vzext.vf2 v10, v8, v0.t
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v10, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i16(<4 x i16> %va, <4 x i1> %m, i32 %evl)
  ret <4 x double> %v
}

define <4 x double> @vuitofp_v4f64_v4i16_unmasked(<4 x i16> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i16_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vzext.vf2 v10, v8
; CHECK-NEXT:    vfwcvt.f.xu.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i16(<4 x i16> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x double> %v
}

declare <4 x double> @llvm.vp.uitofp.v4f64.v4i32(<4 x i32>, <4 x i1>, i32)

define <4 x double> @vuitofp_v4f64_v4i32(<4 x i32> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v10, v8, v0.t
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i32(<4 x i32> %va, <4 x i1> %m, i32 %evl)
  ret <4 x double> %v
}

define <4 x double> @vuitofp_v4f64_v4i32_unmasked(<4 x i32> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i32_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i32(<4 x i32> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x double> %v
}

declare <4 x double> @llvm.vp.uitofp.v4f64.v4i64(<4 x i64>, <4 x i1>, i32)

define <4 x double> @vuitofp_v4f64_v4i64(<4 x i64> %va, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8, v0.t
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i64(<4 x i64> %va, <4 x i1> %m, i32 %evl)
  ret <4 x double> %v
}

define <4 x double> @vuitofp_v4f64_v4i64_unmasked(<4 x i64> %va, i32 zeroext %evl) {
; CHECK-LABEL: vuitofp_v4f64_v4i64_unmasked:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, ta, mu
; CHECK-NEXT:    vfcvt.f.xu.v v8, v8
; CHECK-NEXT:    ret
  %v = call <4 x double> @llvm.vp.uitofp.v4f64.v4i64(<4 x i64> %va, <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer), i32 %evl)
  ret <4 x double> %v
}