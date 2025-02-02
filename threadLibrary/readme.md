Below is an example README document for this project. You can save it as `README.md` (or similar) in your project folder.

---

# User-Level Thread Library Using ucontext and Signals

This project implements a simple user-level thread library in C. The library provides basic thread management functions similar to those found in POSIX threads (pthread), including:

- **`thread_create`** – Create a new thread.
- **`thread_join`** – Wait for a specific thread to finish.
- **`thread_exit`** – Terminate the calling thread.
- **`scheduler_init`** – Initialize the threading system (including the main thread, signal preemption, and the scheduler).

The implementation uses the following concepts:

- **`ucontext` API:**  
  Functions such as `getcontext()`, `makecontext()`, `setcontext()`, and `swapcontext()` are used to capture and switch between thread contexts.  
- **Signals and Timers:**  
  A periodic timer (`setitimer`) is used to send `SIGALRM` signals that preempt the currently running thread by calling the scheduler.
- **Thread Data Structures:**  
  Each thread is represented by a `thread_t` structure which stores the thread’s ID, its execution context, current state, allocated stack, and pointers used to form two linked lists:
  - A **ready queue** (using the `next` field) for threads that are ready to run.
  - A **global list** (using the `all_next` field) for all threads. This global list is used by the join mechanism to “wake up” threads waiting on a thread that has finished.

---

## Files

- **`threads.c`** (or your chosen filename)  
  Contains the full source code for the thread library and sample `main()` functions. Two versions of `main()` are provided via compile-time switches:
  - **EXAMPLE1:** The main thread creates two threads, joins them, and then creates a third thread.
  - **EXAMPLE2:** Demonstrates a thread that creates another thread and then joins it.
  
- **`README.md`**  
  This file.

---

## How It Works

### 1. Scheduler and Signal Initialization

- **`scheduler_init()`**  
  - Sets up a signal handler for `SIGALRM`.  
  - Configures an interval timer so that every 100,000 microseconds (0.1 seconds) the signal handler is invoked.
  - Allocates and initializes the **main thread** (ID 0) and adds it to the global list.
  
  When you run the program, you will see output indicating:
  
  ```
  Initializing scheduler...
  Main thread initialized with ID 0
  ```

### 2. Thread Creation

- **`thread_create(start_routine)`**  
  - Allocates a new `thread_t` structure and a new stack.
  - Uses `getcontext()` to get a context for the new thread, sets the stack, and assigns an exit link.
  - Wraps the user’s start routine in a helper function `thread_wrapper()`, which ensures that once the routine finishes, `thread_exit()` is called.
  - Inserts the new thread into the ready queue and the global list.

### 3. Thread Scheduling

- **`schedule_next()`**  
  - This function is the heart of the thread scheduler.
  - If the current thread is still running, it is reinserted into the ready queue.
  - The scheduler cleans up finished threads and selects the next thread in the ready queue that is in the `THREAD_READY` state.
  - It then switches context to the selected thread using `swapcontext()` or `setcontext()`.
  
- **Preemption:**  
  - The signal handler for `SIGALRM` calls `schedule_next()`, preempting the current thread periodically.

### 4. Thread Exit and Join

- **`thread_exit()`**  
  - Marks the current thread as finished.
  - Frees its stack.
  - Calls `wake_joiners()` to scan the global thread list and mark any thread waiting (joined) on the finishing thread as ready.
  - Calls `schedule_next()` to transfer control to the next thread.
  
- **`thread_join(thread_id)`**  
  - Causes the calling thread to wait until the thread with the given ID finishes.
  - It sets the calling thread’s state to `THREAD_WAITING` and stores the target thread ID.
  - The scheduler will eventually “wake up” the waiting thread when the target thread calls `thread_exit()`.

---

## Compilation

Make sure your system supports the ucontext functions. On Linux, compile with:

```bash
gcc -o mythreads threads.c -Wall -Wextra -D_XOPEN_SOURCE=700
```

You can choose between the provided main routines by defining either `EXAMPLE1` or `EXAMPLE2`. For example, to compile with EXAMPLE1:

```bash
gcc -o mythreads threads.c -Wall -Wextra -D_XOPEN_SOURCE=700 -DEXAMPLE1
```

Or with EXAMPLE2:

```bash
gcc -o mythreads threads.c -Wall -Wextra -D_XOPEN_SOURCE=700 -DEXAMPLE2
```

---

## Running the Program

After compiling, simply run:

```bash
./mythreads
```

You should see output that shows the initialization of the scheduler, thread switches (e.g., "Switching to thread 1"), and thread function iterations. Finally, after all threads finish their execution, the program prints:

```
All threads have finished.
```

---

## Flow Summary

1. **Startup:**  
   - `scheduler_init()` is called to set up signals, the timer, and the main thread.

2. **Creating Threads:**  
   - `thread_create()` is invoked to create new threads with their own stacks and contexts.
   - The new threads are added to both the ready queue and the global list.

3. **Running Threads and Preemption:**  
   - The scheduler (`schedule_next()`) switches between threads either when a thread voluntarily yields (via join) or when the timer signal preempts the running thread.
   - Threads run their designated routines (e.g., printing messages and sleeping).

4. **Thread Completion:**  
   - When a thread completes its routine, it calls `thread_exit()`, marks itself as finished, frees its stack, and wakes up any threads waiting on it.
   - The waiting threads (e.g., in a join call) are reinserted into the ready queue.

5. **Program End:**  
   - The main thread waits for all joined threads to finish.
   - Once all threads are done, the main thread prints the final message and the program exits.

---

## Limitations and Future Improvements

- **Busy-Waiting Join:**  
  The current join implementation uses a blocking mechanism with the scheduler; a more robust version might use event-based signaling.
  
- **Error Handling:**  
  Error checking is minimal. For a production system, additional error handling would be required.
  
- **Resource Management:**  
  The example assumes that once a thread is joined its structure is freed immediately. More sophisticated management might allow threads to be reused or provide a detached state.

---

This project is intended for educational purposes to demonstrate how user-level threading can be implemented using the `ucontext` API and signals for preemption. Enjoy exploring and feel free to extend the library with additional features!

--- 

Feel free to modify this README to suit your project's needs.