#pragma once

namespace ovrc {

// Seconds from an arbitrary fixed origin, using a monotonic clock. Shared by
// all translation units.
double MonotonicSeconds();

} // namespace ovrc
