fetchcontent_declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 12.1.0
)

fetchcontent_makeavailable(fmt)
message(STATUS "Enabling fmt")