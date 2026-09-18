# Thin jlink wrapper. Run on real R 4.5.1 on macOS with an offline fake jev-link:
# CSV round trips, spaces/quotes in arguments, executable fallback, cleanup and errors.
# Not tested against the real Linker, Jev API, Windows, or other R versions.
# Rerun: uv run python tests/wrappers/run.py --r
# Install jlink into an environment on PATH. Prefer jev-link; macOS's jlink is Java.
# Example: jlink(firms, "registry.dta", on=c("name", "city=town"), entity="firm",
#                define="The same firm, despite abbreviations.", left_id="gvkey", right_id="id")

jlink <- function(left, right, on, entity, define = "", left_id = NULL, right_id = NULL,
                  how = "one-to-one", threshold = 0.5, budget = 5) {
    fail <- function(text) stop(paste0("jlink: ", text), call. = FALSE)
    scalar <- function(value, label, empty = FALSE) {
        if (!is.character(value) || length(value) != 1L || is.na(value) ||
            (!empty && !nzchar(trimws(value)))) fail(paste0(label, " must be one text value"))
    }
    if (!is.data.frame(left)) fail("left must be a data frame")
    if (!(is.data.frame(right) || (is.character(right) && length(right) == 1L && !is.na(right)))) {
        fail("right must be a data frame or a table file path")
    }
    if (!is.character(on) || !length(on) || anyNA(on) || any(!nzchar(trimws(on)))) {
        fail("on must name at least one field, for example c('name', 'city=town')")
    }
    scalar(entity, "entity")
    scalar(define, "define", empty = TRUE)
    if (!is.null(left_id)) scalar(left_id, "left_id")
    if (!is.null(right_id)) scalar(right_id, "right_id")
    scalar(how, "how")
    if (!(how %in% c("one-to-one", "many-to-one", "one-to-many", "many-to-many"))) {
        fail("how must be one-to-one, many-to-one, one-to-many or many-to-many")
    }
    if (!is.numeric(threshold) || length(threshold) != 1L || !is.finite(threshold) ||
        threshold < 0 || threshold > 1) fail("threshold must be between 0 and 1")
    if (!is.numeric(budget) || length(budget) != 1L || !is.finite(budget) || budget < 0) {
        fail("budget must be a nonnegative number of US dollars")
    }
    executable <- unname(Sys.which("jev-link"))
    prefix <- character()
    if (!nzchar(executable)) {
        executable <- unname(Sys.which("python3"))
        prefix <- c("-m", "jlink")
    }
    if (!nzchar(executable)) fail("install jev-link or put python3 with jlink installed on PATH")
    work <- tempfile("jlink-")
    if (!dir.create(work)) fail("could not create temporary folder")
    on.exit(unlink(work, recursive = TRUE), add = TRUE)
    left_path <- file.path(work, "left.csv")
    right_path <- file.path(work, "right.csv")
    output <- file.path(work, "links.csv")
    errors <- file.path(work, "stderr.txt")
    log <- file.path(work, "stdout.txt")
    write.csv(left, left_path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
    if (is.data.frame(right)) {
        write.csv(right, right_path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
    } else {
        right_path <- normalizePath(path.expand(right), mustWork = TRUE)
    }
    args <- c(prefix, "link", left_path, right_path, "--entity", entity, "--define", define,
              "--how", how, "--threshold", as.character(threshold), "--budget", as.character(budget),
              "-o", output)
    for (field in on) args <- c(args, "--on", field)
    if (!is.null(left_id)) args <- c(args, "--left-id", left_id)
    if (!is.null(right_id)) args <- c(args, "--right-id", right_id)
    status <- system2(executable, args = shQuote(args), stdout = log, stderr = errors)
    diagnostic <- if (file.exists(errors)) readLines(errors, warn = FALSE) else character()
    if (status != 0L) {
        detail <- diagnostic[startsWith(diagnostic, "jlink:")]
        if (!length(detail)) detail <- paste0("jlink: command failed (exit ", status, "): ",
                                             paste(diagnostic, collapse = " "))
        stop(paste(detail, collapse = "\n"), call. = FALSE)
    }
    if (length(diagnostic)) message(paste(diagnostic, collapse = "\n"))
    if (!file.exists(output)) fail("command succeeded but did not write links.csv")
    # Never infer numeric ID types: an ID such as 00123 must remain 00123.
    result <- read.csv(output, colClasses = "character", check.names = FALSE,
                       na.strings = "", fileEncoding = "UTF-8")
    if (!all(c("left_id", "right_id", "p") %in% names(result))) {
        fail("links.csv must contain left_id, right_id and p")
    }
    for (column in intersect(c("p", "sim", "margin"), names(result))) {
        value <- suppressWarnings(as.numeric(result[[column]]))
        if (any(is.na(value) & !is.na(result[[column]]))) fail(paste0(column, " must be numeric"))
        result[[column]] <- value
    }
    result
}
