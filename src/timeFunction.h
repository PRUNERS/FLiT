#ifndef FUNCTION_TIMER_H
#define FUNCTION_TIMER_H

#include <functional>

namespace flit {

using TimingFunction = std::function<void(void)>;

/** Returns average timing in nanoseconds of the function func
 *
 * The time returned is the best value of all repeated tries.  The time
 * returned is the total looping time divided by the number of loops (so the
 * average runtime over looping).  It is a good idea to loop for fast functions
 * because that amertizes the cost of the looping.
 *
 * @param func - a simple no-effect function to time.  Is executed repeatedly
 * @param loops - how many times to execute func()
 * @param repeats - how often to repeat looping to get the best result
 *
 * @return nanoseconds average runtime of func
 */
int_fast64_t time_function(
    const TimingFunction &func, size_t loops = 1, size_t repeats = 3);

/** Returns average timing in nanoseconds of the function func
 *
 * This is very similar to time_function, but instead of specifying the number
 * of loops, this function determines it automatically.  This is done so that
 * the looping over func() is at least 0.2 seconds in total duration with a
 * maximum looping of one million.
 */
int_fast64_t time_function_autoloop(const TimingFunction &func,
                                    size_t repeats = 3);

} // end of namespace flit

#endif // FUNCTION_TIMER_H
