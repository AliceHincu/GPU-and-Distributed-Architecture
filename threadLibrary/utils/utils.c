#include <time.h>
#include <stdio.h>

/**
 * @brief Suspenda executia pentru un anumit numar de secunde.
 *
 * Aceasta funcție folosește `nanosleep` pentru a astepta durata specificata. 
 * Spre deosebire de functiile `sleep` sau `usleep`, `nanosleep` permite gestionarea
 * intreruperilor cauzate de semnale. Daca un semnal întrerupe somnul, functia 
 * reia nanosleep pentru timpul ramas.
 *
 * @param seconds Durata somnului, exprimată în secunde.
 *                Exemplu: 2.5 pentru 2 secunde și 500 de milisecunde.
 * @return 0
 */
int sleep_seconds(double seconds) {
    int result = 0;

    struct timespec ts_remaining = {
        (time_t)seconds,                
        (long)((seconds - (time_t)seconds) * 1000000000L) 
    };

    do {
        struct timespec ts_sleep = ts_remaining;
        result = nanosleep(&ts_sleep, &ts_remaining);
    } while (result == -1); 

    return result;
}