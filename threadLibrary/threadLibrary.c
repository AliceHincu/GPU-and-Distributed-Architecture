#include "uthread.h"

/* ----------------- Example Program: Proper Read/Write Lock Usage ----------------- */

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

#elif EXAMPLE2

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
    // thread_join(t1);
    thread_join(t2);

    /* Now create a third thread after the first two have finished */
    int t3 = thread_create(thread_function3);
    thread_join(t3);

    printf("All threads have finished.\n");
    return 0;
}

#elif defined(EXAMPLE3)

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

    /* Join on threads t1 and t2; the main thread will be blocked until they finish */
    thread_join(t1);

    /* Now create a third thread after the first two have finished */
    int t3 = thread_create(thread_function3);
    thread_join(t3);

    printf("All threads have finished.\n");
    return 0;
}
#elif EXAMPLE_RWLOCK

int shared_counter = 0;
rwlock_t *my_rwlock;
void writer_thread(void) {
    int id = current_thread->id;
    for (int i = 0; i < 100; i++) {
        /* Acquire exclusive access using the write lock */
        wrlock(my_rwlock);
        int temp = shared_counter;
        sleep_seconds(0.001);
        shared_counter = temp + 1;
        sleep_seconds(0.001);
        rwlock_unlock(my_rwlock);

    }
    printf("Writer %d finished.\n", id);
}

int main() {
    scheduler_init();
    my_rwlock = rwlock_init();
    
    int w1 = thread_create(writer_thread);
    int w2 = thread_create(writer_thread);
    
    thread_join(w1);
    thread_join(w2);
    
    rwlock_destroy(my_rwlock);
    
    printf("Final shared counter = %d (expected 200)\n", shared_counter);
    printf("All threads have finished.\n");
    return 0;
}

#elif EXAMPLE_DEADLOCK
/* Deadlock Example:
   Two mutexes and two threads. Each thread locks one mutex and then attempts to lock the other,
   resulting in a circular wait.
   With the enhanced scheduler, once no threads are ready, deadlock detection is invoked automatically.
*/

mutex_t *mutex1, *mutex2;

void deadlock_thread1(void) {
    int id = current_thread->id;
    printf("Thread %d: Locking mutex1...\n", id);
    mutex_lock(mutex1);
    sleep_seconds(1);
    printf("Thread %d: Attempting to lock mutex2...\n", id);
    mutex_lock(mutex2);
    printf("Thread %d: Acquired both mutexes (should not happen due to deadlock).\n", id);
    mutex_unlock(mutex2);
    mutex_unlock(mutex1);
}

void deadlock_thread2(void) {
    int id = current_thread->id;
    printf("Thread %d: Locking mutex2...\n", id);
    mutex_lock(mutex2);
    sleep_seconds(1);
    printf("Thread %d: Attempting to lock mutex1...\n", id);
    mutex_lock(mutex1);
    printf("Thread %d: Acquired both mutexes (should not happen due to deadlock).\n", id);
    mutex_unlock(mutex1);
    mutex_unlock(mutex2);
}

int main() {
    scheduler_init();

    mutex1 = mutex_init();
    mutex2 = mutex_init();
    
    int t1 = thread_create(deadlock_thread1);
    int t2 = thread_create(deadlock_thread2);
    
    /* Join calls. In deadlock, these will never return normally.
       Instead, the scheduler will detect the deadlock and exit. */
    thread_join(t1);
    thread_join(t2);

    mutex_destroy(mutex1);
    mutex_destroy(mutex2);
    
    printf("All threads have finished.\n");
    return 0;
}

/* ----------------- Example Program: Proper Lock Example ----------------- */
#elif EXAMPLE_LOCK

int shared_counter = 0;
mutex_t* counter_mutex;

void incrementer_thread(void) {
    int id = current_thread->id;
    for (int i = 0; i < 5; i++) {
        printf("Thread %d: Attempting to lock mutex to increment counter...\n", id);
        mutex_lock(counter_mutex);
        printf("Thread %d: Acquired mutex. Counter before = %d\n", id, shared_counter);
        shared_counter++;
        sleep_seconds(0.5);  // simulate some work while holding the lock
        printf("Thread %d: Releasing mutex. Counter now = %d\n", id, shared_counter);
        mutex_unlock(counter_mutex);
        sleep_seconds(0.5);  // simulate work outside of the critical section
    }
}

int main() {
    scheduler_init();

    counter_mutex = mutex_init();
    
    int t1 = thread_create(incrementer_thread);
    int t2 = thread_create(incrementer_thread);
    
    thread_join(t1);
    thread_join(t2);

    mutex_destroy(counter_mutex);
    
    printf("Final shared counter = %d\n", shared_counter);
    printf("All threads have finished.\n");
    return 0;
}

#elif EXAMPLE_NOLOCK

int shared_counter = 0;

void writer_thread(void) {
    int id = current_thread->id;
    for (int i = 0; i < 100; i++) {
        int temp = shared_counter;
        sleep_seconds(0.001);
        shared_counter = temp + 1;
        sleep_seconds(0.001);
    }
    printf("Writer %d finished.\n", id);
}

int main() {
    scheduler_init();
    
    int w1 = thread_create(writer_thread);
    int w2 = thread_create(writer_thread);
    
    thread_join(w1);
    thread_join(w2);
    
    printf("Final shared counter = %d (expected 200)\n", shared_counter);
    printf("All threads have finished.\n");
    return 0;
}


#elif EXAMPLE_RWLOCK_MIX

int shared_counter = 0;
rwlock_t *my_rwlock;

void writer_thread(void) {
    int id = current_thread->id;
    for (int i = 0; i < 100000000; i++) {
        /* Se solicită un write lock pentru acces exclusiv */
        wrlock(my_rwlock);
        shared_counter++;
        rwlock_unlock(my_rwlock);
    }
    printf("Writer %d finished.\n", id);
}

void reader_thread(void) {
    int id = current_thread->id;
    for (int i = 0; i < 100; i++) {
        /* Se solicită un read lock pentru acces concurent */
        rdlock(my_rwlock);
        int value = shared_counter;
        printf("Reader %d read value %d\n", id, value);
        rwlock_unlock(my_rwlock);
    }
    printf("Reader %d finished.\n", id);
}

int main() {
    scheduler_init();
    my_rwlock = rwlock_init();
    
    int w1 = thread_create(writer_thread);
    int r1 = thread_create(reader_thread);
    
    // Așteptăm finalizarea tuturor thread-urilor
    thread_join(w1);
    thread_join(r1);
    
    rwlock_destroy(my_rwlock);
    
    printf("Final shared counter = %d \n", shared_counter);
    printf("All threads have finished.\n");
    return 0;
}

#else
#error "Please define examples to compile the unsynchronized example."
#endif
