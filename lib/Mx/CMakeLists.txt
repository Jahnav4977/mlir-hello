add_mlir_dialect_library(MLIRMx
		MxDialect.cpp
		MxOps.cpp
		LowerToTosa.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Mx

        DEPENDS
        MLIRMxOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
