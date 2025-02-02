#define _XOPEN_SOURCE 700  // for ucontext functions
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>
#include <time.h>

#define INTERVAL_TIME_SIGALRM_MICROSECONDS 100000
#define STACK_SIZE (1024 * 64)

typedef enum { THREAD_READY, THREAD_RUNNING, THREAD_WAITING, THREAD_FINISHED } thread_state_t;

typedef struct thread {
    int id;
    ucontext_t context;
    thread_state_t state;
    void *stack;
    struct thread *next;      // used for the ready queue
    struct thread *all_next;  // used for the global list of all threads
    int target_thread_id;     // if waiting, the id of the thread being joined
} thread_t;

/* Global pointers */
thread_t* current_thread = NULL;   // currently running thread
thread_t* main_thread = NULL;      // main thread
thread_t* ready_queue = NULL;      // ready queue (linked via the "next" pointer)
thread_t* all_threads = NULL;      // global list of all threads (linked via "all_next")
int global_thread_id = 1;          // global thread id counter

/* --- Helper functions --- */

/* Add thread to the tail of the ready queue. */
void add_to_ready_queue(thread_t* thread) {
    thread->next = NULL;
    if (ready_queue == NULL) {
        ready_queue = thread;
    } else {
        thread_t* temp = ready_queue;
        while (temp->next != NULL)
            temp = temp->next;
        temp->next = thread;
    }
}

/* Add a thread to the global list. */
void add_to_all_threads(thread_t* thread) {
    thread->all_next = all_threads;
    all_threads = thread;
}

/* Find a thread in the global list by id (using the all_next field). */
thread_t* find_thread_by_id(int id) {
    thread_t* temp = all_threads;
    while (temp != NULL) {
        if (temp->id == id)
            return temp;
        temp = temp->all_next;
    }
    return NULL;
}

/* Remove finished threads from the ready queue and free their resources.
   (For simplicity, we assume that once a thread is joined its structure is freed.) */
void remove_finished_threads() {
    thread_t* prev = NULL;
    thread_t* curr = ready_queue;
    while (curr != NULL) {
        if (curr->state == THREAD_FINISHED) {
            if (prev == NULL)
                ready_queue = curr->next;
            else
                prev->next = curr->next;
            printf("Cleaning up finished thread %d\n", curr->id);
            /* Freeing the stack was already done when the thread exited */
            free(curr);
            if (prev == NULL)
                curr = ready_queue;
            else
                curr = prev->next;
        } else {
            prev = curr;
            curr = curr->next;
        }
    }
}

/* Sleep for a given number of seconds (supports fractional seconds). */
void sleep_seconds(double seconds) {
    struct timespec ts_remaining = {
        .tv_sec = (time_t)seconds,
        .tv_nsec = (long)((seconds - (time_t)seconds) * 1000000000L)
    };
    while (nanosleep(&ts_remaining, &ts_remaining) == -1)
        ;
}

/* --- Waking Joiners --- */

/* When a thread finishes, wake up any threads that are waiting (joined) on it. */
void wake_joiners(int finished_thread_id) {
    thread_t* temp = all_threads;
    while (temp != NULL) {
        if (temp->state == THREAD_WAITING && temp->target_thread_id == finished_thread_id) {
            temp->state = THREAD_READY;
            temp->target_thread_id = -1;
            add_to_ready_queue(temp);
        }
        temp = temp->all_next;
    }
}

/* --- Scheduler --- */

/* schedule_next() chooses the next READY thread from the ready queue and switches to its context.
   It also “wakes up” waiting threads if the thread they are waiting on has finished.
   If no thread is available, the program exits. */
void schedule_next() {
    /* Do not reinsert a thread that is waiting into the ready queue.
       Only threads that are RUNNING become READY if they voluntarily yield. */
    if (current_thread != NULL && current_thread->state == THREAD_RUNNING) {
        /* Only reinsert if the thread is not waiting. */
        add_to_ready_queue(current_thread);
        current_thread->state = THREAD_READY;
    }

    remove_finished_threads();

    if (ready_queue == NULL) {
        printf("No threads to schedule. Exiting.\n");
        exit(0);
    }

    thread_t *next_thread = ready_queue;
    thread_t *prev_thread = NULL;
    while (next_thread != NULL) {
        /* If a thread is waiting, check if the thread it is waiting for is finished.
           If so, mark it READY. */
        if (next_thread->state == THREAD_WAITING) {
            thread_t* target = find_thread_by_id(next_thread->target_thread_id);
            if (target == NULL || target->state == THREAD_FINISHED) {
                next_thread->state = THREAD_READY;
                next_thread->target_thread_id = -1;
            }
        }
        if (next_thread->state == THREAD_READY)
            break;
        prev_thread = next_thread;
        next_thread = next_thread->next;
    }
    if (next_thread == NULL) {
        printf("No threads left to schedule. Exiting.\n");
        exit(0);
    }
    /* Remove next_thread from the ready queue */
    if (prev_thread != NULL)
        prev_thread->next = next_thread->next;
    else
        ready_queue = next_thread->next;

    next_thread->state = THREAD_RUNNING;
    thread_t *prev = current_thread;
    current_thread = next_thread;

    printf("Switching to thread %d\n", current_thread->id);

    if (prev != NULL) {
        if (swapcontext(&prev->context, &current_thread->context) == -1) {
            perror("swapcontext");
            exit(1);
        }
    } else {
        setcontext(&current_thread->context);
    }
}

/* --- Thread Functions --- */

/* thread_exit() ends the current thread. It marks it finished, frees its stack,
   wakes any joiners, and then calls schedule_next() to run another thread. */
void thread_exit() {
    printf("Thread %d exiting...\n", current_thread->id);
    current_thread->state = THREAD_FINISHED;
    /* Free the stack memory immediately */
    free(current_thread->stack);
    current_thread->stack = NULL;
    /* Wake any threads that are waiting on this thread */
    wake_joiners(current_thread->id);
    schedule_next();
}

/* thread_wrapper() is the function that all new threads start in.
   It calls the user’s start routine and then calls thread_exit(). */
void thread_wrapper(void (*start_routine)(void)) {
    start_routine();
    thread_exit();
}

/* thread_create() creates a new thread that will run start_routine.
   It allocates a new stack and sets up the context. */
int thread_create(void (*start_routine)(void)) {
    thread_t *new_thread = malloc(sizeof(thread_t));
    if (new_thread == NULL) {
        fprintf(stderr, "Failed to allocate memory for thread\n");
        return -1;
    }
    new_thread->stack = malloc(STACK_SIZE);
    if (new_thread->stack == NULL) {
        free(new_thread);
        fprintf(stderr, "Failed to allocate stack for thread\n");
        return -1;
    }
    if (getcontext(&new_thread->context) == -1) {
        perror("getcontext");
        free(new_thread->stack);
        free(new_thread);
        return -1;
    }
    new_thread->id = global_thread_id++;
    new_thread->state = THREAD_READY;
    new_thread->target_thread_id = -1;
    new_thread->context.uc_stack.ss_sp = new_thread->stack;
    new_thread->context.uc_stack.ss_size = STACK_SIZE;
    /* When the thread function returns, control goes to main_thread->context.
       (This is one design choice; you might instead call thread_exit().) */
    new_thread->context.uc_link = &main_thread->context;
    makecontext(&new_thread->context, (void (*)(void))thread_wrapper, 1, start_routine);

    add_to_ready_queue(new_thread);
    add_to_all_threads(new_thread);
    return new_thread->id;
}

/* thread_join() makes the calling thread wait until the thread with thread_id finishes.
   If the target thread is already finished, join returns immediately.
   Otherwise, the calling thread is set to WAITING and schedule_next() is called. */
void thread_join(int thread_id) {
    if (thread_id == 0) {
        fprintf(stderr, "Error: Cannot join the main thread (ID 0).\n");
        return;
    }
    thread_t *target = find_thread_by_id(thread_id);
    if (target == NULL || target->state == THREAD_FINISHED)
        return;
    /* Block the current thread until target finishes */
    current_thread->state = THREAD_WAITING;
    current_thread->target_thread_id = thread_id;
    schedule_next();
}

/* --- Signal Handler and Scheduler Initialization --- */

void signal_handler(int sig) {
    if (sig == SIGALRM) {
        schedule_next();
    }
}

void scheduler_init() {
    printf("Initializing scheduler...\n");
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);

    struct itimerval timer;
    timer.it_value.tv_sec = INTERVAL_TIME_SIGALRM_MICROSECONDS / 1000000;
    timer.it_value.tv_usec = INTERVAL_TIME_SIGALRM_MICROSECONDS;
    timer.it_interval.tv_sec = INTERVAL_TIME_SIGALRM_MICROSECONDS / 1000000;
    timer.it_interval.tv_usec = INTERVAL_TIME_SIGALRM_MICROSECONDS;
    setitimer(ITIMER_REAL, &timer, NULL);

    /* Initialize the main thread */
    main_thread = malloc(sizeof(thread_t));
    if (main_thread == NULL) {
        fprintf(stderr, "Failed to allocate memory for main thread\n");
        exit(EXIT_FAILURE);
    }
    main_thread->id = 0;
    main_thread->state = THREAD_RUNNING;
    main_thread->stack = NULL;  // main thread uses the process’s original stack
    main_thread->target_thread_id = -1;
    if (getcontext(&main_thread->context) == -1) {
        perror("getcontext");
        exit(EXIT_FAILURE);
    }
    current_thread = main_thread;
    add_to_all_threads(main_thread);
    printf("Main thread initialized with ID %d\n", main_thread->id);
}

/* --- Example thread routines --- */


#ifdef EXAMPLE1

void thread_function2() {
    int id = current_thread->id;
    for (int i = 0; i < 5; i++) {
        printf("Thread %d is running (iteration %d)\n", id, i + 1);
        sleep_seconds(1.0);
    }
}

void thread_function1() {
    int id = current_thread->id;
    for (int i = 0; i < 5; i++) {
        printf("Thread %d is running (iteration %d)\n", id, i + 1);
        sleep_seconds(1.5);
    }
}

void thread_function3() {
    int id = current_thread->id;
    printf("Thread %d is running\n", id);
}

/* --- Main --- */

int main() {
    scheduler_init();

    int t1 = thread_create(thread_function1);
    int t2 = thread_create(thread_function2);

    /* Join on threads t1 and t2; the main thread will be blocked until they finish */
    thread_join(t1);
    thread_join(t2);

    /* Now create a third thread after the first two have finished */
    int t3 = thread_create(thread_function3);
    thread_join(t3);

    printf("All threads have finished.\n");
    return 0;
}

#elif defined(EXAMPLE2)

void thread_function2() {
    int id = current_thread->id;
    for (int i = 0; i < 5; i++) {
        printf("Thread %d is running (iteration %d)\n", id, i + 1);
        sleep_seconds(1.0);
    }
}

void thread_function1() {
    int id = current_thread->id;

    int t2 = thread_create(thread_function2);
    thread_join(t2);

    for (int i = 0; i < 5; i++) {
        printf("Thread %d is running (iteration %d)\n", id, i + 1);
        sleep_seconds(1.5);
    }
}

void thread_function3() {
    int id = current_thread->id;
    printf("Thread %d is running\n", id);
}

/* --- Main --- */

int main() {
    scheduler_init();

    int t1 = thread_create(thread_function1);
    // int t2 = thread_create(thread_function2);

    /* Join on threads t1 and t2; the main thread will be blocked until they finish */
    thread_join(t1);
    // thread_join(t2);

    /* Now create a third thread after the first two have finished */
    int t3 = thread_create(thread_function3);
    thread_join(t3);

    printf("All threads have finished.\n");
    return 0;
}

#else
#error "Please define EXAMPLE1 or EXAMPLE2 to choose a main() version."
#endif