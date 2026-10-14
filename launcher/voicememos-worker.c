/*
 * Fixed-command launcher for the voicememos-mcp worker (macOS).
 *
 * Full Disk Access is granted once to this small binary. It starts the worker as a
 * child process; macOS attributes the child's file access to this "responsible"
 * process, so Python can be upgraded without granting access again (the same way
 * programs started from Terminal use Terminal's permissions).
 *
 * The command is compiled in and command-line arguments are ignored, so the binary
 * cannot be used to run anything else with its permissions.
 *
 * Build: cc -O2 -DWORKER_PYTHON='"/path/to/venv/bin/python"' -o voicememos-worker voicememos-worker.c
 */
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>

#ifndef WORKER_PYTHON
#error "define WORKER_PYTHON as the worker interpreter path"
#endif

extern char **environ;

int main(void) {
    char *const argv[] = {WORKER_PYTHON, "-m", "voicememos_mcp", "sync", NULL};
    pid_t pid;
    int status;
    int rc = posix_spawn(&pid, WORKER_PYTHON, NULL, NULL, argv, environ);
    if (rc != 0) {
        fprintf(stderr, "voicememos-worker: cannot start %s (error %d)\n", WORKER_PYTHON, rc);
        return 127;
    }
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("voicememos-worker: waitpid");
            return 1;
        }
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 128 + WTERMSIG(status);
}
