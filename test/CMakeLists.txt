configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MX_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        mx-opt
        )

add_lit_testsuite(check-mx "Running the mx regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MX_TEST_DEPENDS}
        )
set_target_properties(check-mx PROPERTIES FOLDER "Tests")

add_lit_testsuites(MX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MX_TEST_DEPENDS})
