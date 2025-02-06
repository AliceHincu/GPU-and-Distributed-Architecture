#ifndef UTHREAD_H
#define UTHREAD_H

#define _XOPEN_SOURCE 700  // for ucontext functions
#include <ucontext.h>
#include <stdbool.h>

/* Timer interval and stack size */
#define INTERVAL_TIME_SIGALRM_MICROSECONDS 100000
#define STACK_SIZE (1024 * 64)

/* Thread states */
typedef enum { 
    THREAD_READY, 
    THREAD_RUNNING, 
    THREAD_WAITING, 
    THREAD_FINISHED 
} thread_state_t;

/* Waiting type for synchronization objects */
enum { 
    WAIT_NONE = 0, 
    WAIT_READ, 
    WAIT_WRITE 
};

/* Forward declaration of thread_t */
typedef struct thread thread_t;

/* Thread structure */
struct thread {
    int id;
    ucontext_t context;
    thread_state_t state;
    void *stack;
    struct thread *next;      // for the ready queue
    struct thread *all_next;  // for the global list of threads
    int target_thread_id;     // for join operations (if nonnegative, the thread being joined)
    void *waiting_on;         // pointer to the sync object (mutex_t* or rwlock_t*) the thread is waiting on
    int waiting_for;          // WAIT_NONE, WAIT_READ, or WAIT_WRITE
};

/* Global thread management variables */
extern thread_t* current_thread;   // currently running thread
extern thread_t* main_thread;      // main thread (ID 0)
extern thread_t* ready_queue;      // ready queue (linked via 'next')
extern thread_t* all_threads;      // global list of all threads (linked via 'all_next')
extern int global_thread_id;       // global thread id counter

/* ----------------- Mutex Definition ----------------- */

/* Full definition of mutex_t */
typedef struct mutex {
    bool locked;
    thread_t *owner;
    thread_t *wait_head;
    thread_t *wait_tail;
} mutex_t;

/* ----------------- Read/Write Lock Definition ----------------- */

/* Full definition of rwlock_t */
typedef struct rwlock {
    bool writer_active;      // true if a writer holds the lock
    int reader_count;        // number of active readers
    thread_t *writer_owner;  // pointer to the writer (if active)
    thread_t *active_reader; // representative active reader (for deadlock detection)
    thread_t *wait_head;     // waiting queue head
    thread_t *wait_tail;     // waiting queue tail
} rwlock_t;

/* ----------------- Function Prototypes ----------------- */

/* Thread management and scheduling */
void scheduler_init(void);
int thread_create(void (*start_routine)(void));
void thread_join(int thread_id);
void thread_exit(void);
void schedule_next(void);
void wake_joiners(int finished_id);
void sleep_seconds(double seconds);

/* Mutex functions */
mutex_t *mutex_init(void);
void mutex_destroy(mutex_t *m);
void mutex_lock(mutex_t *m);
void mutex_unlock(mutex_t *m);

/* Read/Write lock functions */
rwlock_t *rwlock_init(void);
void rwlock_destroy(rwlock_t *rw);
void rdlock(rwlock_t *rw);
void wrlock(rwlock_t *rw);
void rwlock_unlock(rwlock_t *rw);

/* Deadlock detection */
bool detect_deadlock(void);

#endif /* UTHREAD_H */
