/*
 * Vendored from the fda-launcher template (keep in sync).
 *
 * fda-launcher: a fixed-command launcher that can hold a macOS privacy permission
 * (for example Full Disk Access) on behalf of one background job.
 *
 * macOS ties privacy permissions to the exact executable. Scripts cannot hold them
 * (the interpreter would need the permission, and then every script would have it),
 * and interpreter binaries change with every update. This launcher is a tiny,
 * stable binary: grant it the permission once, and the command it starts as a
 * child process uses that permission, the same way programs started from Terminal
 * use Terminal's permissions.
 *
 * The command is compiled in (LAUNCH_ARGV) and command-line arguments are ignored,
 * so the binary cannot be used to run anything else. Build one launcher per job
 * with build-launcher.sh, which handles quoting and only rebuilds when needed.
 */
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef LAUNCH_ARGV
#error "define LAUNCH_ARGV, e.g. -DLAUNCH_ARGV='\"/abs/path/program\",\"arg\"' (use build-launcher.sh)"
#endif

extern char **environ;

static char *const launch_argv[] = {LAUNCH_ARGV, NULL};
static volatile pid_t child = 0;

static void forward(int sig) {
    if (child > 0) {
        kill(child, sig);
    }
}

int main(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = forward;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    pid_t pid;
    int rc = posix_spawn(&pid, launch_argv[0], NULL, NULL, launch_argv, environ);
    if (rc != 0) {
        fprintf(stderr, "fda-launcher: cannot start %s: %s\n", launch_argv[0], strerror(rc));
        return 127;
    }
    child = pid;

    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("fda-launcher: waitpid");
            return 1;
        }
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 128 + WTERMSIG(status);
}
