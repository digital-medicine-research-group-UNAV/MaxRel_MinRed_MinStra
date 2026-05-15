#!/usr/bin/env Rscript

# download_bnlearn_repository.R
#
# Descarga redes bayesianas del repositorio de bnlearn:
# https://www.bnlearn.com/bnrepository/
#
# Uso:
#   Rscript download_bnlearn_repository.R
#   Rscript download_bnlearn_repository.R --out_dir ./networks --formats bif --overwrite true
#   Rscript download_bnlearn_repository.R --formats bif,net --keep_gz true
#
parse_args <- function(args) {
  opts <- list(
    out_dir = default_out_dir(),
    formats = "bif",
    overwrite = FALSE,
    manifest = TRUE,
    verbose = TRUE,
    keep_gz = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (startsWith(arg, "--")) {
      key <- sub("^--", "", arg)
      if (grepl("=", key, fixed = TRUE)) {
        parts <- strsplit(key, "=", fixed = TRUE)[[1]]
        key <- parts[1]
        value <- paste(parts[-1], collapse = "=")
      } else if (i < length(args) && !startsWith(args[[i + 1]], "--")) {
        value <- args[[i + 1]]
        i <- i + 1
      } else {
        value <- "TRUE"
      }

      if (key %in% c("overwrite", "manifest", "verbose", "keep_gz")) {
        opts[[key]] <- tolower(value) %in% c("true", "1", "yes", "y")
      } else {
        opts[[key]] <- value
      }
    }
    i <- i + 1
  }

  opts$formats <- unique(trimws(unlist(strsplit(opts$formats, ","))))
  opts
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    return(NA_character_)
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE)
}

default_out_dir <- function() {
  this_script <- script_path()
  if (is.na(this_script)) {
    return(file.path(getwd(), "networks"))
  }
  file.path(dirname(this_script), "networks")
}

log_msg <- function(..., verbose = TRUE) {
  if (isTRUE(verbose)) {
    cat(sprintf(...), "\n")
  }
}

normalize_url <- function(url) {
  url <- sub("#.*$", "", url)
  url <- sub("[?].*$", "", url)
  url
}

resolve_url <- function(base_url, href) {
  if (is.na(href) || !nzchar(href)) return(NA_character_)
  if (grepl("^https?://", href, ignore.case = TRUE)) return(normalize_url(href))
  if (startsWith(href, "/")) return(normalize_url(paste0("https://www.bnlearn.com", href)))
  base_dir <- sub("[^/]*$", "", base_url)
  normalize_url(paste0(base_dir, href))
}

read_hrefs <- function(url) {
  html <- paste(readLines(url, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  matches <- gregexpr("href\\s*=\\s*['\"][^'\"]+['\"]", html, perl = TRUE)
  tokens <- regmatches(html, matches)[[1]]
  if (length(tokens) == 0) return(character())

  hrefs <- sub("^href\\s*=\\s*['\"]", "", tokens, perl = TRUE)
  hrefs <- sub("['\"]$", "", hrefs, perl = TRUE)
  hrefs <- hrefs[nzchar(hrefs)]
  unique(hrefs)
}

is_repository_page <- function(url, index_url) {
  if (!startsWith(url, index_url)) return(FALSE)
  grepl("/$", url) || grepl("\\.html$", url, ignore.case = TRUE)
}

crawl_repository_pages <- function(index_url, verbose = TRUE) {
  queue <- c(index_url)
  visited <- character()
  pages <- character()

  while (length(queue) > 0) {
    current <- queue[[1]]
    queue <- queue[-1]

    if (current %in% visited) next
    visited <- c(visited, current)

    hrefs <- tryCatch(
      read_hrefs(current),
      error = function(e) {
        log_msg("[warn] No se pudo leer %s (%s)", current, conditionMessage(e), verbose = verbose)
        character()
      }
    )
    pages <- c(pages, current)

    if (length(hrefs) == 0) next

    absolute_links <- unique(vapply(hrefs, resolve_url, FUN.VALUE = character(1), base_url = current))
    absolute_links <- absolute_links[!is.na(absolute_links)]

    candidate_pages <- absolute_links[vapply(absolute_links, is_repository_page, logical(1), index_url = index_url)]
    new_pages <- setdiff(candidate_pages, visited)
    if (length(new_pages) > 0) {
      queue <- c(queue, new_pages)
    }
  }

  unique(pages)
}

extract_format <- function(file_url) {
  file_name <- tolower(basename(file_url))
  file_name <- sub("\\.gz$", "", file_name)
  tools::file_ext(file_name)
}

discover_download_links <- function(pages, formats, index_url, verbose = TRUE) {
  pattern <- paste0("\\.(", paste(formats, collapse = "|"), ")(\\.gz)?$")
  all_links <- character()

  for (page in pages) {
    hrefs <- tryCatch(
      read_hrefs(page),
      error = function(e) {
        log_msg("[warn] No se pudieron extraer enlaces de %s (%s)", page, conditionMessage(e), verbose = verbose)
        character()
      }
    )
    if (length(hrefs) == 0) next

    absolute_links <- unique(vapply(hrefs, resolve_url, FUN.VALUE = character(1), base_url = page))
    absolute_links <- absolute_links[!is.na(absolute_links)]
    absolute_links <- absolute_links[startsWith(absolute_links, index_url)]

    file_links <- absolute_links[grepl(pattern, absolute_links, ignore.case = TRUE)]
    all_links <- c(all_links, file_links)
  }

  unique(all_links)
}

decompress_gzip <- function(src_gz, dest_file) {
  con_in <- gzfile(src_gz, "rb")
  con_out <- file(dest_file, "wb")
  on.exit({
    close(con_in)
    close(con_out)
  }, add = TRUE)

  repeat {
    chunk <- readBin(con_in, what = "raw", n = 1024 * 1024)
    if (length(chunk) == 0) break
    writeBin(chunk, con_out)
  }
}

build_destfile <- function(out_dir, file_url, keep_gz = FALSE) {
  file_name <- basename(file_url)
  if (!keep_gz) {
    file_name <- sub("\\.gz$", "", file_name, ignore.case = TRUE)
  }
  file.path(out_dir, file_name)
}

safe_download <- function(url, destfile, overwrite = FALSE, keep_gz = FALSE, verbose = TRUE) {
  if (file.exists(destfile) && !overwrite) {
    log_msg("[skip] %s", destfile, verbose = verbose)
    return(list(status = "skipped", ok = TRUE, size = file.info(destfile)$size, destfile = destfile))
  }

  dir.create(dirname(destfile), recursive = TRUE, showWarnings = FALSE)

  tmp_file <- tempfile(pattern = "bnlearn_", fileext = ".tmp")
  ok <- FALSE
  err <- NULL

  tryCatch({
    utils::download.file(url, destfile = tmp_file, mode = "wb", quiet = !verbose, method = "libcurl")
    if (!keep_gz && grepl("\\.gz$", url, ignore.case = TRUE)) {
      decompress_gzip(tmp_file, destfile)
    } else {
      ok_copy <- file.copy(tmp_file, destfile, overwrite = TRUE)
      if (!ok_copy) stop("No se pudo copiar el archivo temporal al destino.")
    }
    ok <- TRUE
  }, error = function(e) {
    err <<- conditionMessage(e)
  }, finally = {
    if (file.exists(tmp_file)) unlink(tmp_file)
  })

  if (!ok) {
    return(list(status = "error", ok = FALSE, error = err, size = NA_integer_, destfile = destfile))
  }

  list(status = "downloaded", ok = TRUE, size = file.info(destfile)$size, destfile = destfile)
}

download_bnlearn_repository <- function(out_dir,
                                        formats = c("bif"),
                                        overwrite = FALSE,
                                        manifest = TRUE,
                                        verbose = TRUE,
                                        keep_gz = FALSE) {
  index_url <- "https://www.bnlearn.com/bnrepository/"
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  valid_formats <- c("bif", "dsc", "net", "rda", "rds")
  formats <- tolower(formats)
  bad_formats <- setdiff(formats, valid_formats)
  if (length(bad_formats) > 0) {
    stop(sprintf(
      "Formato(s) no soportado(s): %s. Usa solo: %s",
      paste(bad_formats, collapse = ", "),
      paste(valid_formats, collapse = ", ")
    ), call. = FALSE)
  }

  log_msg("Repositorio índice: %s", index_url, verbose = verbose)
  log_msg("Directorio de salida: %s", normalizePath(out_dir, winslash = "/", mustWork = FALSE), verbose = verbose)
  log_msg("Formatos: %s", paste(formats, collapse = ", "), verbose = verbose)
  log_msg("keep_gz: %s", keep_gz, verbose = verbose)

  pages <- crawl_repository_pages(index_url, verbose = verbose)
  log_msg("Páginas detectadas: %d", length(pages), verbose = verbose)

  links <- discover_download_links(pages, formats = formats, index_url = index_url, verbose = verbose)
  log_msg("Ficheros detectados para descarga: %d", length(links), verbose = verbose)

  manifest_rows <- list()
  row_id <- 1L

  for (file_url in links) {
    destfile <- build_destfile(out_dir, file_url, keep_gz = keep_gz)
    result <- safe_download(
      file_url,
      destfile = destfile,
      overwrite = overwrite,
      keep_gz = keep_gz,
      verbose = verbose
    )

    manifest_rows[[row_id]] <- data.frame(
      file_name = basename(file_url),
      format = extract_format(file_url),
      source_url = file_url,
      destfile = result$destfile,
      status = result$status,
      size_bytes = if (!is.null(result$size)) result$size else NA_integer_,
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }

  if (length(manifest_rows) > 0) {
    manifest_df <- do.call(rbind, manifest_rows)
  } else {
    manifest_df <- data.frame(
      file_name = character(),
      format = character(),
      source_url = character(),
      destfile = character(),
      status = character(),
      size_bytes = integer(),
      stringsAsFactors = FALSE
    )
  }

  if (isTRUE(manifest)) {
    manifest_path <- file.path(out_dir, "manifest_bnlearn_downloads.csv")
    utils::write.csv(manifest_df, manifest_path, row.names = FALSE)
    log_msg("Manifest guardado en: %s", manifest_path, verbose = verbose)
  }

  invisible(manifest_df)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  download_bnlearn_repository(
    out_dir = opts$out_dir,
    formats = opts$formats,
    overwrite = opts$overwrite,
    manifest = opts$manifest,
    verbose = opts$verbose,
    keep_gz = opts$keep_gz
  )
}

if (sys.nframe() == 0) {
  main()
}
