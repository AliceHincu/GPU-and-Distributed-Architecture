#include "uthread.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

/* ----------------- Global Variables ----------------- */

thread_t* current_thread = NULL;   // currently running thread
thread_t* main_thread = NULL;      // main thread (ID 0)
thread_t* ready_queue = NULL;      // ready queue (linked via 'next')
thread_t* all_threads = NULL;      // global list of all threads (linked via 'all_next')
int global_thread_id = 1;          // global thread id counter

/* ----------------- Helper Functions ----------------- */

void add_to_ready_queue(thread_t* thread) {
    // add to end of queue
    thread->next = NULL;
    if (ready_queue == NULL)
        ready_queue = thread;
    else {
        thread_t* temp = ready_queue;
        while (temp->next)
            temp = temp->next;
        temp->next = thread;
    }
}

void add_to_all_threads(thread_t* thread) {
    // add to beginning (order not important)
    thread->all_next = all_threads;
    all_threads = thread;
}

thread_t* find_thread_by_id(int id) {
    thread_t* temp = all_threads;
    while (temp) {
        if (temp->id == id)
            return temp;
        temp = temp->all_next;
    }
    return NULL;
}

void remove_finished_threads(void) {
    thread_t* prev = NULL;
    thread_t* curr = ready_queue;
    while (curr) {
        if (curr->state == THREAD_FINISHED) {
            if (prev)
                prev->next = curr->next;
            else
                ready_queue = curr->next;
            printf("Cleaning up finished thread %d\n", curr->id);
            free(curr);
            if (prev)
                curr = prev->next;
            else
                curr = ready_queue;
        } else {
            prev = curr;
            curr = curr->next;
        }
    }
}

void sleep_seconds(double seconds) {
    // uses nanosleep instead of sleep/usleep because it allows handling intreruptions caused by signals.
    // in case of interruptions, nanosleep reloads from remaining time.
    struct timespec ts = { .tv_sec = (time_t)seconds,
                           .tv_nsec = (long)((seconds - (time_t)seconds) * 1000000000L) };
    while (nanosleep(&ts, &ts) == -1);
}

/* ----------------- Mutex Implementation ----------------- */

mutex_t *mutex_init(void) {
    mutex_t *m = malloc(sizeof(mutex_t));
    if (m == NULL) {
        perror("malloc mutex_init");
        exit(EXIT_FAILURE);
    }
    m->locked = false;
    m->owner = NULL;
    m->wait_head = m->wait_tail = NULL;
    return m;
}

void mutex_destroy(mutex_t *m) {
    if (m->locked)
        fprintf(stderr, "Error: Cannot destroy a locked mutex.\n");
    free(m);
}

void mutex_lock(mutex_t *m) {
    if (!m->locked) {
        m->locked = true;
        m->owner = current_thread;
        return;
    }

    // make current thread wait
    current_thread->state = THREAD_WAITING;
    current_thread->waiting_on = (void *)m;
    current_thread->waiting_for = WAIT_NONE; 
    current_thread->next = NULL;

    // add it to waiting queue 
    if (m->wait_tail == NULL) {
        m->wait_head = current_thread;
        m->wait_tail = current_thread;
    } else {
        m->wait_tail->next = current_thread;
        m->wait_tail = current_thread;
    }

    // yield execution to next thread
    schedule_next();
}

void mutex_unlock(mutex_t *m) {
    if (m->owner != current_thread) {
        fprintf(stderr, "Error: Thread %d does not own the mutex.\n", thread_self());
        return;
    }
    if (m->wait_head != NULL) {
        // give mutex to first thread in waiting queue
        thread_t *t = m->wait_head;
        m->wait_head = t->next;
        if (m->wait_head == NULL)
            m->wait_tail = NULL;
        m->owner = t;
        t->waiting_on = NULL;
        t->waiting_for = WAIT_NONE;
        t->state = THREAD_READY;
        t->next = NULL;
        add_to_ready_queue(t);
    } else {
        // free mutex
        m->locked = false;
        m->owner = NULL;
    }
}

/* ----------------- Read/Write Lock Implementation ----------------- */

rwlock_t *rwlock_init(void) {
    rwlock_t *rw = malloc(sizeof(rwlock_t));
    rw->writer_active = false;
    rw->reader_count = 0;
    rw->writer_owner = NULL;
    rw->active_reader = NULL;
    rw->wait_head = rw->wait_tail = NULL;
    return rw;
}

void rwlock_destroy(rwlock_t *rw) {
    if (rw->writer_active || rw->reader_count > 0 || rw->wait_head != NULL)
        fprintf(stderr, "Error: Cannot destroy a locked rwlock.\n");
    free(rw);
}

void rdlock(rwlock_t *rw) {
    // avoid write-write and read-write conflict
    if (!rw->writer_active && rw->wait_head == NULL) {
        rw->reader_count++;
        if (rw->reader_count == 1)
            rw->active_reader = current_thread;
        return;
    }
    
    // make current thread wait
    current_thread->state = THREAD_WAITING;
    current_thread->waiting_on = (void *)rw;
    current_thread->waiting_for = WAIT_READ;
    current_thread->next = NULL;

    // add it to waiting queue 
    if (rw->wait_tail == NULL) {
        rw->wait_head = current_thread;
        rw->wait_tail = current_thread;
    } else {
        rw->wait_tail->next = current_thread;
        rw->wait_tail = current_thread;
    }

    // yield execution to next thread
    schedule_next();

    // after the wait is done, get the lock
    rw->reader_count++;
    if (rw->reader_count == 1)
        rw->active_reader = current_thread;
    current_thread->waiting_on = NULL;
    current_thread->waiting_for = WAIT_NONE;
}

void wrlock(rwlock_t *rw) {
    // avoid write-read and write-write conflict
    if (rw->reader_count == 0 && !rw->writer_active) {
        rw->writer_active = true;
        rw->writer_owner = current_thread;
        return;
    }

    // make current thread wait
    current_thread->state = THREAD_WAITING;
    current_thread->waiting_on = (void *)rw;
    current_thread->waiting_for = WAIT_WRITE;
    current_thread->next = NULL;

    // add it to waiting queue 
    if (rw->wait_tail == NULL) {
        rw->wait_head = current_thread;
        rw->wait_tail = current_thread;
    } else {
        rw->wait_tail->next = current_thread;
        rw->wait_tail = current_thread;
    }

    // yield execution to next thread
    schedule_next();

    // after the wait is done, get the lock
    rw->writer_active = true;
    rw->writer_owner = current_thread;
    current_thread->waiting_on = NULL;
    current_thread->waiting_for = WAIT_NONE;
}

void rwlock_unlock(rwlock_t *rw) {
    if (rw->writer_active && rw->writer_owner == current_thread) {
        // release write lock
        rw->writer_active = false;
        rw->writer_owner = NULL;
    } else if (rw->reader_count > 0) {
        // release read lock
        rw->reader_count--;
        if (rw->reader_count == 0)
            rw->active_reader = NULL;
    } else {
        // current thread does not own the lock
        fprintf(stderr, "Error: rwlock unlock: current thread does not hold the lock.\n");
        return;
    }

    // check if you can wake up the next waiting thread from the waiting queue
    if (rw->wait_head != NULL) {
        thread_t *t = rw->wait_head;
        if (t->waiting_for == WAIT_WRITE) {
            // next waiting thread is a writer
            if (rw->reader_count == 0 && !rw->writer_active) {
                // wake it up if there are no conflicts
                rw->writer_active = true;
                rw->writer_owner = t;
                rw->wait_head = t->next;
                if (rw->wait_head == NULL)
                    rw->wait_tail = NULL;
                t->state = THREAD_READY;
                add_to_ready_queue(t);
            }
        } else if (t->waiting_for == WAIT_READ) {
            // next waiting thread is a reader
            while (rw->wait_head && rw->wait_head->waiting_for == WAIT_READ) {
                // wake it up if there are no conflicts
                thread_t *r = rw->wait_head;
                rw->reader_count++;
                if (rw->reader_count == 1)
                    rw->active_reader = r;
                rw->wait_head = r->next;
                if (rw->wait_head == NULL)
                    rw->wait_tail = NULL;
                r->state = THREAD_READY;
                add_to_ready_queue(r);
            }
        }
        // otherwise wait for next unlock
    }
}

/* ----------------- Wake Joiners ----------------- */

void wake_joiners(int finished_id) {
    thread_t *t = all_threads;
    while (t != NULL) {
        if (t->state == THREAD_WAITING && t->target_thread_id == finished_id) {
            t->state = THREAD_READY;
            t->target_thread_id = -1;
            add_to_ready_queue(t);
        }
        t = t->all_next;
    }
}

/* ----------------- Deadlock Detection ----------------- */

bool detect_deadlock(void) {
    printf("Deadlock detection invoked:\n");
    bool deadlock_found = false;
    thread_t *t = all_threads;

    while (t != NULL) {
        // for each thread, check if there is a waiting cycle 
        if (t->waiting_on && t->state == THREAD_WAITING) {
            thread_t *start = t;
            thread_t *chain = t;
            bool cycle = false;

            // start parsing waiting chain
            while (chain && chain->waiting_on) {
                thread_t *next = NULL;
            
                if (chain->waiting_for == WAIT_NONE) {
                    // if the thread waits for a mutex, get the owner
                    mutex_t *m = (mutex_t *)chain->waiting_on;
                    next = m->owner;
                } else if (chain->waiting_for == WAIT_READ) {
                    // if thread waits for a read lock, get the writer owner it there is one
                    rwlock_t *rw = (rwlock_t *)chain->waiting_on;
                    next = (rw->writer_active ? rw->writer_owner : NULL);
                } else if (chain->waiting_for == WAIT_WRITE) {
                    // if thread waits for a write lock, get either the writer owner or the active reader
                    rwlock_t *rw = (rwlock_t *)chain->waiting_on;
                    if (rw->writer_active)
                        next = rw->writer_owner;
                    else if (rw->reader_count > 0)
                        next = rw->active_reader;
                }

                // if the cycle ended => deadlock
                if (next == start) {
                    cycle = true;
                    break;
                }

                // if there is no next or we reach the same element, chain is over
                if (next == NULL || next == chain)
                    break;

                chain = next;
            }

            if (cycle) {
                printf("Deadlock detected involving thread %d\n", t->id);
                deadlock_found = true;
            }
        }
        t = t->all_next;
    }
    return deadlock_found;
}

/* ----------------- Scheduler ----------------- */

void signal_handler(int sig) {
    if (sig == SIGALRM)
        schedule_next();
}

void scheduler_init(void) {
    printf("Initializing scheduler...\n");
    // associate signal with schedule_next
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask); // don't block other signals
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, NULL);
    
    // use timer to trigger signal automatically
    struct itimerval timer;
    timer.it_value.tv_sec = INTERVAL_TIME_SIGALRM_MICROSECONDS / 1000000;
    timer.it_value.tv_usec = INTERVAL_TIME_SIGALRM_MICROSECONDS;
    timer.it_interval.tv_sec = INTERVAL_TIME_SIGALRM_MICROSECONDS / 1000000;
    timer.it_interval.tv_usec = INTERVAL_TIME_SIGALRM_MICROSECONDS;
    setitimer(ITIMER_REAL, &timer, NULL);
    
    // create main thread
    main_thread = malloc(sizeof(thread_t));
    if (!main_thread) {
        fprintf(stderr, "Failed to allocate memory for main thread\n");
        exit(EXIT_FAILURE);
    }
    main_thread->id = 0;
    main_thread->state = THREAD_RUNNING;
    main_thread->stack = NULL;  // main thread uses the process's original stack
    main_thread->target_thread_id = -1;
    main_thread->waiting_on = NULL;
    main_thread->waiting_for = WAIT_NONE;
    if (getcontext(&main_thread->context) == -1) {
        // capture current context of main thread
        // Later on, when a thread finishes its execution or if there are no other threads to run, the scheduler can switch back to this captured context.
        perror("getcontext");
        exit(EXIT_FAILURE);
    }
    current_thread = main_thread;
    add_to_all_threads(main_thread);
    printf("Main thread initialized with ID %d\n", main_thread->id);
}


void schedule_next(void) {
    if (current_thread != NULL && current_thread->state == THREAD_RUNNING) {
        add_to_ready_queue(current_thread);
        current_thread->state = THREAD_READY;
    }
    remove_finished_threads();

    if (ready_queue == NULL) {
        // check if threads are stuck waiting (deadlock). If not, program is done.
        bool waiting_found = false;
        thread_t *temp = all_threads;
        while (temp != NULL) {
            if (temp->state == THREAD_WAITING) {
                waiting_found = true;
                break;
            }
            temp = temp->all_next;
        }
        if (waiting_found) {
            printf("No threads in the ready queue, but some threads are still waiting.\n");
            printf("Invoking deadlock detection...\n");
            if(detect_deadlock()) {
                printf("Exiting due to deadlock.\n");
                exit(1);
            } else {
                printf("No deadlock detected; waiting for threads to become ready...\n");
                return;
            }
        } else {
            printf("No threads to schedule. Exiting.\n");
            exit(0);
        }
    }

    // get first ready thread
    thread_t *next_thread = ready_queue;
    ready_queue = next_thread->next;
    next_thread->state = THREAD_RUNNING;

    // the current thread becomes the next found ready thread
    thread_t *prev = current_thread;
    current_thread = next_thread;
    printf("Switching to thread %d\n", current_thread->id);

    // swap context from previous current thread to next new ready current thread
    if (prev != NULL) {
        if (swapcontext(&prev->context, &current_thread->context) == -1) {
            perror("swapcontext");
            exit(1);
        }
    } else {
        setcontext(&current_thread->context);
    }
}

/* ----------------- Thread Functions ----------------- */

int thread_self(void) {
    return current_thread->id;
}

void thread_exit(void) {
    printf("Thread %d exiting...\n", thread_self());
    current_thread->state = THREAD_FINISHED;
    wake_joiners(thread_self());
    free(current_thread->stack);
    current_thread->stack = NULL;
    schedule_next();
}

void thread_wrapper(void (*start_routine)(void)) {
    start_routine();
    thread_exit();
}

int thread_create(void (*start_routine)(void)) {
    thread_t *new_thread = malloc(sizeof(thread_t));
    if (!new_thread) {
        fprintf(stderr, "Failed to allocate memory for thread\n");
        return -1;
    }
    new_thread->stack = malloc(STACK_SIZE);
    if (!new_thread->stack) {
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
    new_thread->waiting_on = NULL;
    new_thread->waiting_for = WAIT_NONE;
    new_thread->context.uc_stack.ss_sp = new_thread->stack;
    new_thread->context.uc_stack.ss_size = STACK_SIZE;
    new_thread->context.uc_link = &main_thread->context;
    makecontext(&new_thread->context, (void (*)(void))thread_wrapper, 1, start_routine);
    add_to_ready_queue(new_thread);
    add_to_all_threads(new_thread);
    return new_thread->id;
}

void thread_join(int thread_id) {
    if (thread_id == 0) {
        fprintf(stderr, "Error: Cannot join the main thread (ID 0).\n");
        return;
    }
    thread_t *target = find_thread_by_id(thread_id);
    if (!target || target->state == THREAD_FINISHED)
        return;
    current_thread->state = THREAD_WAITING;
    current_thread->target_thread_id = thread_id;
    schedule_next();
}