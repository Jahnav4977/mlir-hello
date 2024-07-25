; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare void @free(ptr)

declare i32 @printf(ptr, ...)

declare ptr @malloc(i64)

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 2, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 6) to i64))
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 2, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 3, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 3, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 4, 1
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %18 = getelementptr double, ptr %17, i64 0
  store double 1.000000e+00, ptr %18, align 8
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %20 = getelementptr double, ptr %19, i64 1
  store double 2.000000e+00, ptr %20, align 8
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %22 = getelementptr double, ptr %21, i64 2
  store double 3.000000e+00, ptr %22, align 8
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %24 = getelementptr double, ptr %23, i64 3
  store double 4.000000e+00, ptr %24, align 8
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %26 = getelementptr double, ptr %25, i64 4
  store double 5.000000e+00, ptr %26, align 8
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 1
  %28 = getelementptr double, ptr %27, i64 5
  store double 6.000000e+00, ptr %28, align 8
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %30 = getelementptr double, ptr %29, i64 0
  store double 1.000000e+00, ptr %30, align 8
  %31 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %32 = getelementptr double, ptr %31, i64 1
  store double 2.000000e+00, ptr %32, align 8
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %34 = getelementptr double, ptr %33, i64 2
  store double 3.000000e+00, ptr %34, align 8
  %35 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %36 = getelementptr double, ptr %35, i64 3
  store double 4.000000e+00, ptr %36, align 8
  %37 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %38 = getelementptr double, ptr %37, i64 4
  store double 5.000000e+00, ptr %38, align 8
  %39 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %40 = getelementptr double, ptr %39, i64 5
  store double 6.000000e+00, ptr %40, align 8
  br label %41

41:                                               ; preds = %56, %0
  %42 = phi i64 [ 0, %0 ], [ %58, %56 ]
  %43 = icmp slt i64 %42, 2
  br i1 %43, label %44, label %59

44:                                               ; preds = %41
  br label %45

45:                                               ; preds = %48, %44
  %46 = phi i64 [ 0, %44 ], [ %55, %48 ]
  %47 = icmp slt i64 %46, 3
  br i1 %47, label %48, label %56

48:                                               ; preds = %45
  %49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %50 = mul i64 %42, 3
  %51 = add i64 %50, %46
  %52 = getelementptr double, ptr %49, i64 %51
  %53 = load double, ptr %52, align 8
  %54 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %53)
  %55 = add i64 %46, 1
  br label %45

56:                                               ; preds = %45
  %57 = call i32 (ptr, ...) @printf(ptr @nl)
  %58 = add i64 %42, 1
  br label %41

59:                                               ; preds = %41
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0
  call void @free(ptr %60)
  %61 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  call void @free(ptr %61)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

