// Fails the day every supported compiler has C++26 deleted-function reasons,
// so the SKIPPY_DELETE fallback can be dropped (house rule, docs/TYPE_DESIGN.md).
#include "skippy/poison.h"

#include <gtest/gtest.h>

#include <version>

TEST(Toolchain, SkippyDeleteFallbackCanGo) {
    EXPECT_TRUE(__cpp_deleted_function >= 202403L) << "SKIPPY_DELETE fallback can be dropped";
}
