#define _XOPEN_SOURCE 700 // to get rid of not defined error

#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h> // https://pubs.opengroup.org/onlinepubs/7908799/xsh/ucontext.h.html
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>

#define INTERVAL_TIME_SIGALRM_MICROSECONDS 100000
#define STACK_SIZE (1024 * 64)

typedef enum { THREAD_READY, THREAD_RUNNING, THREAD_FINISHED } thread_state_t;

typedef struct thread {
    int id;
    ucontext_t context;
    thread_state_t state;
    void *stack;
    struct thread *next;
} thread_t;

thread_t* current_thread = NULL; // Thread-ul care rulează acum
thread_t* main_thread = NULL;
thread_t* ready_queue = NULL;    // Lista de thread-uri gata să ruleze
int global_thread_id = 1; // ID global pentru thread-uri


void add_to_ready_queue(thread_t* thread) {
    if (ready_queue == NULL) {
        ready_queue = thread;
    } else {
        thread_t* temp = ready_queue;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = thread;
    }
    thread->next = NULL;
}

void remove_finished_threads() {
    thread_t* prev = NULL;
    thread_t* curr = ready_queue;

    while (curr != NULL) {
        if (curr->state == THREAD_FINISHED) {
            if (prev == NULL) {
                ready_queue = curr->next;
            } else {
                prev->next = curr->next;
            }

            printf("Cleaning up finished thread %d\n", curr->id);
            free(curr->stack);
            free(curr);

            if (prev == NULL) {
                curr = ready_queue;
            } else {
                curr = prev->next;
            }
        } else {
            prev = curr;
            curr = curr->next;
        }
    }
}

void schedule_next() {
    if (current_thread != NULL && current_thread->state == THREAD_RUNNING) {
        current_thread->state = THREAD_READY;
        add_to_ready_queue(current_thread);
    }

    remove_finished_threads();

    if (ready_queue == NULL) {
        printf("No threads to schedule. Exiting.\n");
        exit(0);
    }

    thread_t* next_thread = ready_queue;
    ready_queue = ready_queue->next;
    next_thread->state = THREAD_RUNNING;

    printf("Switching to thread %d\n", next_thread->id);
    thread_t* prev_thread = current_thread;
    current_thread = next_thread;

    if (prev_thread != NULL) {
        if (swapcontext(&prev_thread->context, &current_thread->context) == -1) {
            perror("swapcontext");
            exit(1);
        }
    } else {
        setcontext(&current_thread->context);
    }
}

void thread_wrapper(thread_t *thread, void (*start_routine)(void)) {
    start_routine();

    thread->state = THREAD_FINISHED;
    printf("Thread %d finished. Switching context...\n", thread->id);

    schedule_next();
}

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
    new_thread->context.uc_stack.ss_sp = new_thread->stack;
    new_thread->context.uc_stack.ss_size = STACK_SIZE;
    new_thread->context.uc_link = &main_thread->context;
    makecontext(&new_thread->context, (void (*)(void))thread_wrapper, 2, new_thread, start_routine);

    add_to_ready_queue(new_thread);
    return 0;
}

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

    main_thread = malloc(sizeof(thread_t));
    if (main_thread == NULL) {
        fprintf(stderr, "Failed to allocate memory for main thread\n");
        exit(EXIT_FAILURE);
    }
    main_thread->id = 0;
    main_thread->state = THREAD_RUNNING;
    main_thread->stack = NULL;
    if (getcontext(&(main_thread->context)) == -1) {
        printf("Error while getting context...exiting\n");
        exit(EXIT_FAILURE);
    }
    current_thread = main_thread;
    printf("Main thread initialized with ID %d\n", main_thread->id);
}

void thread_function1() {
    for (int i = 0; i < 5; i++) {
        printf("Thread 1 is running (iteration %d)\n", i + 1);
        usleep(500000);
    }
}

void thread_function2() {
    for (int i = 0; i < 5; i++) {
        printf("Thread 2 is running (iteration %d)\n", i + 1);
        usleep(500000);
    }
}

int main() {
    scheduler_init();

    thread_create(thread_function1);
    thread_create(thread_function2);

    while (1) {
        pause(); // Menține programul activ
    }

    return 0;
}
