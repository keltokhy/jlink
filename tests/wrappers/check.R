args <- commandArgs(trailingOnly = TRUE)
source(args[1])
cat(R.version.string, "\n")
right <- args[2]
left <- data.frame(firm_id = c("00123", "00456"), name = c("Acme & Sons", "Other firm"),
                   city = c("Boston", "New York"))
original <- left
run <- function(right, define = Sys.getenv("JLINK_TEST_DEFINE")) {
    jlink(left, right, on = c("name", "city=town"), entity = "firm", define = define,
          left_id = "firm_id", right_id = "record_id", threshold = 0.8, budget = 0)
}
links <- run(right)
stopifnot(identical(links$left_id, "00123"), identical(links$right_id, "00007"),
          identical(links$p, 0.98), identical(left, original))
links <- run(data.frame(record_id = "00007", name = "Acme", town = "Boston"))
stopifnot(identical(links$left_id, "00123"), identical(left, original))
problem <- tryCatch(run(right, "FAIL"), error = function(e) conditionMessage(e))
stopifnot(grepl("jlink: registry.dta: column 'name' is missing", problem, fixed = TRUE),
          identical(left, original))
relation <- jlink(left, right, on = c("name", "city=town"), define = Sys.getenv("JLINK_TEST_DEFINE"),
                  style = "rule", left_id = "firm_id", right_id = "record_id")
stopifnot(identical(relation$left_id, "00123"), identical(left, original))
for (bad in list(list(style = "rule"), list(style = "relation", entity = "firm"), list())) {
    problem <- tryCatch(do.call(jlink, c(list(left, right, on = "name"), bad)),
                        error = function(e) conditionMessage(e))
    stopifnot(startsWith(problem, "jlink: "))
}
saved <- file.path(getwd(), "run folder's files")
kept <- jlink(left, right, on = c("name", "city=town"), entity = "firm", left_id = "firm_id",
              right_id = "record_id", run_dir = saved)
stopifnot(identical(kept$left_id, "00123"), file.exists(file.path(saved, "settings.json")))
empty <- run(right, "EMPTY")
stopifnot(nrow(empty) == 0L, identical(names(empty), names(links)), identical(left, original))
cat("R_WRAPPER_OK\n")
