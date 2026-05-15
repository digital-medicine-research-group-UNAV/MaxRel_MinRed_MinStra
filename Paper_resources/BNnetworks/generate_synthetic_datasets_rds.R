#!/usr/bin/env Rscript

# generate_synthetic_datasets_rds.R
#
# Genera datasets sintéticos desde redes bayesianas en formato .rds (bn.fit).
#
# Uso:
#   Rscript generate_synthetic_datasets_rds.R \
#     --input_dir ./networks \
#     --output_dir ./synthetic_datasets_rds \
#     --n_samples 250,1000,5000 \
#     --seed 42 \
#     --n_replications 1 \
#     --target_type discrete_or_continuous \
#     --min_class_balance 0.08 \
#     --continuous_n_bins auto \
#     --continuous_n_bins_min 3 \
#     --continuous_n_bins_max 10 \
#     --continuous_min_bin_prop 0.08 \
#     --discretize_continuous_targets true \
#     --balance_eval_samples 10000
#
# Dependencias:
#   install.packages("bnlearn")
#   install.packages("jsonlite")

SUMMARY_COLUMNS <- c(
  "network_name",
  "target",
  "n_nodes",
  "n_parameters",
  "n_parents",
  "n_children",
  "n_spouses",
  "mb_size",
  "parents",
  "children",
  "spouses",
  "markov_boundary"
)

MAX_TARGETS_DEFAULT <- 10L
MIN_CLASS_BALANCE_DEFAULT <- 0.08
TARGET_TYPE_DEFAULT <- "discrete_or_continuous"
CONTINUOUS_N_BINS_DEFAULT <- "auto"
CONTINUOUS_N_BINS_MIN_DEFAULT <- 3L
CONTINUOUS_N_BINS_MAX_DEFAULT <- 10L
CONTINUOUS_MIN_BIN_PROP_DEFAULT <- 0.08
DISCRETIZE_CONTINUOUS_TARGETS_DEFAULT <- TRUE
BALANCE_EVAL_SAMPLES_DEFAULT <- 10000L

log_msg <- function(fmt, ..., verbose = TRUE) {
  if (isTRUE(verbose)) {
    cat(sprintf(fmt, ...), "\n")
  }
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    return(NA_character_)
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE)
}

default_input_dir <- function() {
  this_script <- script_path()
  if (is.na(this_script)) {
    return(file.path(getwd(), "networks"))
  }
  file.path(dirname(this_script), "networks")
}

default_output_dir <- function() {
  this_script <- script_path()
  if (is.na(this_script)) {
    return(file.path(getwd(), "synthetic_datasets_rds"))
  }
  file.path(dirname(this_script), "synthetic_datasets_rds")
}

ensure_dependencies <- function() {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop(
      "Falta la dependencia 'jsonlite'. Instálala con install.packages('jsonlite').",
      call. = FALSE
    )
  }
  if (!requireNamespace("bnlearn", quietly = TRUE)) {
    stop(
      "Falta la dependencia 'bnlearn'. Instálala con install.packages('bnlearn').",
      call. = FALSE
    )
  }
}

parse_bool <- function(x) {
  tolower(x) %in% c("true", "1", "yes", "y")
}

parse_args <- function(args) {
  opts <- list(
    input_dir = default_input_dir(),
    network = NA_character_,
    output_dir = default_output_dir(),
    n_samples = "250,1000,5000",
    seed = 42L,
    n_replications = 1L,
    max_targets = MAX_TARGETS_DEFAULT,
    target_type = TARGET_TYPE_DEFAULT,
    min_class_balance = MIN_CLASS_BALANCE_DEFAULT,
    continuous_n_bins = CONTINUOUS_N_BINS_DEFAULT,
    continuous_n_bins_min = CONTINUOUS_N_BINS_MIN_DEFAULT,
    continuous_n_bins_max = CONTINUOUS_N_BINS_MAX_DEFAULT,
    continuous_min_bin_prop = CONTINUOUS_MIN_BIN_PROP_DEFAULT,
    discretize_continuous_targets = DISCRETIZE_CONTINUOUS_TARGETS_DEFAULT,
    balance_eval_samples = BALANCE_EVAL_SAMPLES_DEFAULT,
    verbose = TRUE
  )

  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (!startsWith(arg, "--")) {
      i <- i + 1L
      next
    }

    key <- sub("^--", "", arg)
    if (grepl("=", key, fixed = TRUE)) {
      parts <- strsplit(key, "=", fixed = TRUE)[[1]]
      key <- parts[1]
      value <- paste(parts[-1], collapse = "=")
    } else if (i < length(args) && !startsWith(args[[i + 1L]], "--")) {
      value <- args[[i + 1L]]
      i <- i + 1L
    } else {
      value <- "TRUE"
    }

    if (key %in% c("verbose", "discretize_continuous_targets")) {
      opts[[key]] <- parse_bool(value)
    } else {
      opts[[key]] <- value
    }
    i <- i + 1L
  }

  opts$input_dir <- normalizePath(opts$input_dir, winslash = "/", mustWork = FALSE)
  opts$output_dir <- normalizePath(opts$output_dir, winslash = "/", mustWork = FALSE)
  opts$seed <- as.integer(opts$seed)
  opts$n_replications <- as.integer(opts$n_replications)
  opts$max_targets <- as.integer(opts$max_targets)
  opts$target_type <- as.character(opts$target_type)
  opts$min_class_balance <- as.numeric(opts$min_class_balance)
  opts$continuous_n_bins <- as.character(opts$continuous_n_bins)
  opts$continuous_n_bins_min <- as.integer(opts$continuous_n_bins_min)
  opts$continuous_n_bins_max <- as.integer(opts$continuous_n_bins_max)
  opts$continuous_min_bin_prop <- as.numeric(opts$continuous_min_bin_prop)
  opts$balance_eval_samples <- as.integer(opts$balance_eval_samples)
  opts$n_samples <- parse_sample_sizes(opts$n_samples)
  opts
}

parse_sample_sizes <- function(raw_value) {
  chunks <- trimws(unlist(strsplit(raw_value, ",")))
  chunks <- chunks[nzchar(chunks)]
  if (length(chunks) == 0) {
    stop("No se han especificado valores válidos para --n_samples.", call. = FALSE)
  }
  parsed <- as.integer(chunks)
  if (any(is.na(parsed)) || any(parsed <= 0L)) {
    stop("Todos los valores de --n_samples deben ser enteros > 0.", call. = FALSE)
  }
  sort(unique(parsed))
}

discover_network_files <- function(input_dir) {
  if (!dir.exists(input_dir)) {
    stop(sprintf("El directorio de entrada no existe: %s", input_dir), call. = FALSE)
  }
  files <- list.files(input_dir, pattern = "\\.rds$", full.names = TRUE, ignore.case = TRUE)
  files <- sort(files)
  if (length(files) == 0) {
    stop(sprintf("No se encontraron archivos .rds en %s", input_dir), call. = FALSE)
  }
  files
}

network_base_name <- function(file_path) {
  tools::file_path_sans_ext(basename(file_path))
}

filter_networks <- function(network_files, requested_network = NA_character_) {
  if (is.na(requested_network) || !nzchar(requested_network)) {
    return(network_files)
  }
  req <- tolower(requested_network)
  keep <- vapply(
    network_files,
    function(path) {
      name <- tolower(basename(path))
      stem <- tolower(network_base_name(path))
      identical(name, req) || identical(stem, req)
    },
    logical(1)
  )
  selected <- network_files[keep]
  if (length(selected) == 0) {
    available <- paste(vapply(network_files, network_base_name, character(1)), collapse = ", ")
    stop(
      sprintf("No se encontró la red '%s'. Disponibles: %s", requested_network, available),
      call. = FALSE
    )
  }
  selected
}

load_network <- function(network_file) {
  obj <- readRDS(network_file)
  if (!inherits(obj, "bn.fit")) {
    stop(sprintf("El archivo no contiene un objeto 'bn.fit': %s", network_file), call. = FALSE)
  }
  obj
}

get_node_names <- function(model) {
  sort(names(model))
}

get_parents <- function(model, node) {
  sort(unique(as.character(model[[node]]$parents)))
}

get_children <- function(model, node) {
  sort(unique(as.character(model[[node]]$children)))
}

get_spouses <- function(model, node) {
  node_children <- get_children(model, node)
  spouses <- character()
  for (child in node_children) {
    child_parents <- get_parents(model, child)
    spouses <- c(spouses, setdiff(child_parents, node))
  }
  sort(unique(spouses))
}

get_markov_boundary <- function(parents, children, spouses) {
  sort(unique(c(parents, children, spouses)))
}

is_valid_target <- function(parents, children, spouses) {
  (length(parents) >= 1L) && (length(children) >= 1L) && (length(spouses) >= 1L)
}

is_discrete_target <- function(model, node) {
  identical(class(model[[node]])[1], "bn.fit.dnode")
}

is_continuous_target <- function(model, node) {
  node_class <- class(model[[node]])[1]
  identical(node_class, "bn.fit.gnode") || identical(node_class, "bn.fit.cgnode")
}

meets_discrete_balance <- function(sampled_df, target, min_class_balance) {
  values <- sampled_df[[target]]
  if (is.null(values)) return(FALSE)

  class_freq <- table(values)
  if (length(class_freq) < 2L) {
    return(FALSE)
  }

  class_probs <- as.numeric(class_freq) / sum(class_freq)
  min(class_probs) >= min_class_balance
}

compute_quantile_bins <- function(values, n_bins) {
  probs <- seq(0, 1, length.out = n_bins + 1L)
  quantiles <- as.numeric(stats::quantile(values, probs = probs, na.rm = TRUE, names = FALSE))
  if (length(quantiles) <= 2L) {
    return(NULL)
  }
  inner <- unique(quantiles[2:(length(quantiles) - 1L)])
  breaks <- c(-Inf, inner, Inf)
  if (length(breaks) <= 2L) {
    pretty_breaks <- unique(as.numeric(pretty(values, n = n_bins)))
    if (length(pretty_breaks) > 2L) {
      min_v <- min(values, na.rm = TRUE)
      max_v <- max(values, na.rm = TRUE)
      inner_pretty <- pretty_breaks[pretty_breaks > min_v & pretty_breaks < max_v]
      breaks <- c(-Inf, unique(inner_pretty), Inf)
    } else {
      breaks <- c(-Inf, Inf)
    }
  }
  if (length(breaks) <= 2L) {
    return(NULL)
  }
  breaks
}

get_continuous_binning <- function(values,
                                   requested_n_bins,
                                   bins_min,
                                   bins_max,
                                   min_bin_prop) {
  if (is.null(values)) return(NULL)
  values <- as.numeric(values)
  values <- values[is.finite(values)]
  if (length(values) < 2L || length(unique(values)) < 2L) {
    return(NULL)
  }

  evaluate_k <- function(k) {
    breaks <- compute_quantile_bins(values, n_bins = k)
    if (is.null(breaks)) return(NULL)
    bins <- cut(values, breaks = breaks, include.lowest = TRUE, right = TRUE, ordered_result = FALSE)
    bin_freq <- table(bins)
    if (length(bin_freq) < 2L) return(NULL)
    bin_probs <- as.numeric(bin_freq) / sum(bin_freq)
    min_prop <- min(bin_probs)
    if (min_prop < min_bin_prop) return(NULL)
    list(n_bins = length(breaks) - 1L, breaks = breaks, min_prop = min_prop)
  }

  req <- tolower(trimws(requested_n_bins))
  if (identical(req, "auto")) {
    k_from <- bins_max
    k_to <- bins_min
    if (k_from < k_to) {
      tmp <- k_from
      k_from <- k_to
      k_to <- tmp
    }
    for (k in seq(from = k_from, to = k_to, by = -1L)) {
      candidate <- evaluate_k(k)
      if (!is.null(candidate)) {
        return(candidate)
      }
    }
    return(NULL)
  }

  k <- suppressWarnings(as.integer(req))
  if (is.na(k) || k < 2L) {
    return(NULL)
  }
  evaluate_k(k)
}

discretize_continuous_target <- function(values, breaks) {
  bins <- cut(values, breaks = breaks, include.lowest = TRUE, right = TRUE, ordered_result = TRUE)
  idx <- as.integer(bins)
  if (any(is.na(idx))) {
    stop("No se pudo discretizar el target continuo (se obtuvieron bins NA).", call. = FALSE)
  }
  as.integer(idx - 1L)
}

is_supported_target_type <- function(model, node, target_type) {
  if (identical(target_type, "discrete_only")) {
    return(is_discrete_target(model, node))
  }
  if (identical(target_type, "discrete_or_continuous")) {
    return(is_discrete_target(model, node) || is_continuous_target(model, node))
  }
  FALSE
}

find_valid_targets <- function(model,
                               sampled_for_balance,
                               target_type,
                               min_class_balance,
                               continuous_n_bins,
                               continuous_n_bins_min,
                               continuous_n_bins_max,
                               continuous_min_bin_prop,
                               max_targets = MAX_TARGETS_DEFAULT) {
  node_names <- get_node_names(model)
  rows <- list()
  idx <- 1L

  for (node in node_names) {
    parents <- get_parents(model, node)
    children <- get_children(model, node)
    spouses <- get_spouses(model, node)
    if (!is_valid_target(parents, children, spouses)) {
      next
    }
    if (!is_supported_target_type(model, node, target_type)) {
      next
    }
    if (is_discrete_target(model, node) &&
        !meets_discrete_balance(sampled_for_balance, node, min_class_balance)) {
      next
    }
    if (is_continuous_target(model, node) &&
        is.null(get_continuous_binning(
          sampled_for_balance[[node]],
          requested_n_bins = continuous_n_bins,
          bins_min = continuous_n_bins_min,
          bins_max = continuous_n_bins_max,
          min_bin_prop = continuous_min_bin_prop
        ))) {
      next
    }
    continuous_binning <- NULL
    if (is_continuous_target(model, node)) {
      continuous_binning <- get_continuous_binning(
        sampled_for_balance[[node]],
        requested_n_bins = continuous_n_bins,
        bins_min = continuous_n_bins_min,
        bins_max = continuous_n_bins_max,
        min_bin_prop = continuous_min_bin_prop
      )
    }
    rows[[idx]] <- list(
      target = node,
      parents = parents,
      children = children,
      spouses = spouses,
      markov_boundary = get_markov_boundary(parents, children, spouses),
      continuous_binning = continuous_binning
    )
    idx <- idx + 1L
  }

  if (length(rows) == 0L) {
    return(rows)
  }

  # Ya están ordenados por nombre del nodo (node_names sorted).
  if (length(rows) > max_targets) {
    rows <- rows[seq_len(max_targets)]
  }
  rows
}

is_dag_from_model <- function(model) {
  nodes <- names(model)
  indegree <- setNames(integer(length(nodes)), nodes)
  adjacency <- setNames(vector("list", length(nodes)), nodes)
  for (n in nodes) adjacency[[n]] <- character()

  for (child in nodes) {
    parents <- get_parents(model, child)
    for (parent in parents) {
      adjacency[[parent]] <- unique(c(adjacency[[parent]], child))
      indegree[[child]] <- indegree[[child]] + 1L
    }
  }

  queue <- names(indegree[indegree == 0L])
  visited <- 0L
  while (length(queue) > 0L) {
    node <- queue[[1]]
    queue <- queue[-1]
    visited <- visited + 1L
    for (neigh in adjacency[[node]]) {
      indegree[[neigh]] <- indegree[[neigh]] - 1L
      if (indegree[[neigh]] == 0L) {
        queue <- c(queue, neigh)
      }
    }
  }
  visited == length(nodes)
}

compute_n_parameters <- function(model) {
  total <- 0
  unknown <- FALSE

  for (node in names(model)) {
    node_obj <- model[[node]]
    node_class <- class(node_obj)[1]

    if (identical(node_class, "bn.fit.dnode")) {
      dims <- dim(node_obj$prob)
      if (is.null(dims)) {
        unknown <- TRUE
        next
      }
      variable_card <- dims[1]
      parent_configs <- if (length(dims) == 1) 1L else prod(dims[-1])
      total <- total + (variable_card - 1L) * parent_configs
      next
    }

    if (identical(node_class, "bn.fit.gnode")) {
      total <- total + length(node_obj$coefficients) + 1L
      next
    }

    if (identical(node_class, "bn.fit.cgnode")) {
      coeff <- node_obj$coefficients
      if (is.null(dim(coeff))) {
        total <- total + length(coeff) + 1L
      } else {
        # Por configuración discreta: intercepto + betas gaussianas + sd
        total <- total + ncol(coeff) * (nrow(coeff) + 1L)
      }
      next
    }

    unknown <- TRUE
  }

  if (unknown) {
    return(NA_integer_)
  }
  as.integer(total)
}

sanitize_name <- function(x) {
  gsub("[^A-Za-z0-9_.-]+", "_", x)
}

serialize_list <- function(x) {
  jsonlite::toJSON(unname(x), auto_unbox = TRUE)
}

sample_dataset <- function(model, target, n_samples, seed_rep, target_breaks = NULL) {
  set.seed(seed_rep)
  sampled <- bnlearn::rbn(model, n = n_samples)

  if (!target %in% colnames(sampled)) {
    stop(sprintf("El target '%s' no aparece en el dataset sampleado.", target), call. = FALSE)
  }

  if (!is.null(target_breaks)) {
    sampled[[target]] <- discretize_continuous_target(sampled[[target]], breaks = target_breaks)
  }

  ordered_cols <- c(target, sort(setdiff(colnames(sampled), target)))
  sampled[, ordered_cols, drop = FALSE]
}

validate_dataset_shape <- function(df, n_samples, n_expected_columns) {
  if (nrow(df) != n_samples) {
    stop(
      sprintf("Dataset con %d filas; se esperaban %d.", nrow(df), n_samples),
      call. = FALSE
    )
  }
  if (ncol(df) != n_expected_columns) {
    stop(
      sprintf("Dataset con %d columnas; se esperaban %d.", ncol(df), n_expected_columns),
      call. = FALSE
    )
  }
}

save_dataset <- function(df, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(df, output_path, row.names = FALSE)
}

save_summary <- function(summary_rows, summary_path) {
  dir.create(dirname(summary_path), recursive = TRUE, showWarnings = FALSE)

  if (length(summary_rows) == 0L) {
    empty_df <- as.data.frame(setNames(replicate(length(SUMMARY_COLUMNS), character(0), simplify = FALSE), SUMMARY_COLUMNS))
    utils::write.csv(empty_df, summary_path, row.names = FALSE)
    return(invisible(NULL))
  }

  df <- do.call(rbind, lapply(summary_rows, as.data.frame, stringsAsFactors = FALSE))
  df <- df[, SUMMARY_COLUMNS, drop = FALSE]
  utils::write.csv(df, summary_path, row.names = FALSE)
}

save_metadata <- function(metadata, metadata_path) {
  dir.create(dirname(metadata_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(metadata, metadata_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
}

build_summary_rows <- function(network_name, n_nodes, n_parameters, targets) {
  rows <- list()
  if (length(targets) == 0L) return(rows)

  for (i in seq_along(targets)) {
    target <- targets[[i]]
    rows[[i]] <- list(
      network_name = network_name,
      target = target$target,
      n_nodes = n_nodes,
      n_parameters = if (is.na(n_parameters)) NA else n_parameters,
      n_parents = length(target$parents),
      n_children = length(target$children),
      n_spouses = length(target$spouses),
      mb_size = length(target$markov_boundary),
      parents = serialize_list(target$parents),
      children = serialize_list(target$children),
      spouses = serialize_list(target$spouses),
      markov_boundary = serialize_list(target$markov_boundary)
    )
  }
  rows
}

process_network <- function(network_file,
                            output_root,
                            sample_sizes,
                            seed,
                            n_replications,
                            max_targets,
                            target_type,
                            min_class_balance,
                            continuous_n_bins,
                            continuous_n_bins_min,
                            continuous_n_bins_max,
                            continuous_min_bin_prop,
                            discretize_continuous_targets,
                            balance_eval_samples,
                            verbose = TRUE) {
  network_name <- network_base_name(network_file)
  network_dir <- file.path(output_root, network_name)
  datasets_dir <- file.path(network_dir, "datasets")
  summary_path <- file.path(network_dir, "summary_targets.csv")
  metadata_path <- file.path(network_dir, "network_metadata.json")

  dir.create(network_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(datasets_dir, recursive = TRUE, showWarnings = FALSE)

  log_msg("[INFO] Cargando red: %s", network_file, verbose = verbose)
  model <- load_network(network_file)

  if (!is_dag_from_model(model)) {
    stop(sprintf("La red '%s' no es un DAG válido.", network_name), call. = FALSE)
  }

  node_names <- get_node_names(model)
  n_nodes <- length(node_names)
  n_parameters <- compute_n_parameters(model)

  set.seed(as.integer(seed))
  sampled_for_balance <- bnlearn::rbn(model, n = balance_eval_samples)

  log_msg("[INFO] Red '%s' con %d nodos", network_name, n_nodes, verbose = verbose)
  log_msg(
    "[INFO] Filtro targets: target_type=%s, min_class_balance=%.4f, continuous_n_bins=%s, continuous_n_bins_min=%d, continuous_n_bins_max=%d, continuous_min_bin_prop=%.4f, discretize_continuous_targets=%s, eval_samples=%d",
    target_type, min_class_balance, continuous_n_bins, continuous_n_bins_min, continuous_n_bins_max, continuous_min_bin_prop, tolower(as.character(discretize_continuous_targets)), balance_eval_samples, verbose = verbose
  )

  all_valid_targets <- find_valid_targets(
    model = model,
    sampled_for_balance = sampled_for_balance,
    target_type = target_type,
    min_class_balance = min_class_balance,
    continuous_n_bins = continuous_n_bins,
    continuous_n_bins_min = continuous_n_bins_min,
    continuous_n_bins_max = continuous_n_bins_max,
    continuous_min_bin_prop = continuous_min_bin_prop,
    max_targets = 1000000L
  )
  n_all_valid <- length(all_valid_targets)
  targets <- all_valid_targets
  if (length(targets) > max_targets) {
    targets <- targets[seq_len(max_targets)]
    log_msg(
      "[INFO] Targets válidos en '%s': %d (limitados a los primeros %d).",
      network_name, n_all_valid, max_targets, verbose = verbose
    )
  } else {
    log_msg("[INFO] Targets válidos en '%s': %d", network_name, n_all_valid, verbose = verbose)
  }

  summary_rows <- build_summary_rows(
    network_name = network_name,
    n_nodes = n_nodes,
    n_parameters = n_parameters,
    targets = targets
  )
  save_summary(summary_rows, summary_path)

  target_breaks_map <- list()
  if (isTRUE(discretize_continuous_targets)) {
    for (target in targets) {
      target_name <- target$target
      if (!is_continuous_target(model, target_name)) {
        next
      }
      breaks <- target$continuous_binning$breaks
      if (is.null(breaks)) {
        stop(
          sprintf("No se pudieron construir bins para discretizar target continuo '%s'.", target_name),
          call. = FALSE
        )
      }
      target_breaks_map[[target_name]] <- breaks
    }
  }

  generated <- 0L
  for (target_idx in seq_along(targets)) {
    target <- targets[[target_idx]]

    if (!is_valid_target(target$parents, target$children, target$spouses)) {
      stop(sprintf("Target inválido tras validación: %s", target$target), call. = FALSE)
    }

    safe_target <- sanitize_name(target$target)
    for (n_samp in sample_sizes) {
      for (rep_id in seq_len(n_replications)) {
        index_target_zero_based <- target_idx - 1L
        seed_rep <- as.integer(seed + index_target_zero_based * 1000L + rep_id)

        df <- sample_dataset(
          model = model,
          target = target$target,
          n_samples = n_samp,
          seed_rep = seed_rep,
          target_breaks = target_breaks_map[[target$target]]
        )
        validate_dataset_shape(df, n_samples = n_samp, n_expected_columns = n_nodes)

        file_name <- sprintf(
          "%s__target_%s__n_%d__rep_%d.csv",
          network_name, safe_target, n_samp, rep_id
        )
        save_dataset(df, file.path(datasets_dir, file_name))
        generated <- generated + 1L
      }
    }
  }

  metadata <- list(
    network_name = network_name,
    source_file = normalizePath(network_file, winslash = "/", mustWork = FALSE),
    n_nodes = n_nodes,
    node_names = node_names,
    valid_targets = vapply(targets, function(x) x$target, character(1)),
    n_valid_targets = length(targets),
    n_valid_targets_before_limit = n_all_valid,
    target_limit = max_targets,
    sampling_method = "bnlearn::rbn",
    seed = seed,
    target_type = target_type,
    min_class_balance = min_class_balance,
    continuous_n_bins = continuous_n_bins,
    continuous_n_bins_min = continuous_n_bins_min,
    continuous_n_bins_max = continuous_n_bins_max,
    continuous_min_bin_prop = continuous_min_bin_prop,
    discretize_continuous_targets = discretize_continuous_targets,
    balance_eval_samples = balance_eval_samples,
    n_samples = sample_sizes,
    n_replications = n_replications,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  )
  save_metadata(metadata, metadata_path)

  log_msg(
    "[INFO] Red '%s' completada. Datasets generados: %d. Salida: %s",
    network_name, generated, network_dir, verbose = verbose
  )
}

main <- function() {
  ensure_dependencies()
  opts <- parse_args(commandArgs(trailingOnly = TRUE))

  if (is.na(opts$seed)) {
    stop("--seed debe ser un entero.", call. = FALSE)
  }
  if (is.na(opts$n_replications) || opts$n_replications < 1L) {
    stop("--n_replications debe ser >= 1.", call. = FALSE)
  }
  if (is.na(opts$max_targets) || opts$max_targets < 1L) {
    stop("--max_targets debe ser >= 1.", call. = FALSE)
  }
  if (!opts$target_type %in% c("discrete_only", "discrete_or_continuous")) {
    stop("--target_type debe ser 'discrete_only' o 'discrete_or_continuous'.", call. = FALSE)
  }
  if (is.na(opts$min_class_balance) || opts$min_class_balance < 0 || opts$min_class_balance >= 1) {
    stop("--min_class_balance debe estar en [0, 1).", call. = FALSE)
  }
  if (is.na(opts$continuous_n_bins_min) || opts$continuous_n_bins_min < 2L) {
    stop("--continuous_n_bins_min debe ser >= 2.", call. = FALSE)
  }
  if (is.na(opts$continuous_n_bins_max) || opts$continuous_n_bins_max < opts$continuous_n_bins_min) {
    stop("--continuous_n_bins_max debe ser >= --continuous_n_bins_min.", call. = FALSE)
  }
  continuous_n_bins_raw <- tolower(trimws(opts$continuous_n_bins))
  if (!identical(continuous_n_bins_raw, "auto")) {
    parsed_cont_bins <- suppressWarnings(as.integer(continuous_n_bins_raw))
    if (is.na(parsed_cont_bins) || parsed_cont_bins < 2L) {
      stop("--continuous_n_bins debe ser 'auto' o un entero >= 2.", call. = FALSE)
    }
  }
  if (is.na(opts$continuous_min_bin_prop) || opts$continuous_min_bin_prop < 0 || opts$continuous_min_bin_prop >= 1) {
    stop("--continuous_min_bin_prop debe estar en [0, 1).", call. = FALSE)
  }
  if (is.na(opts$balance_eval_samples) || opts$balance_eval_samples < 100L) {
    stop("--balance_eval_samples debe ser >= 100.", call. = FALSE)
  }

  network_files <- discover_network_files(opts$input_dir)
  selected <- filter_networks(network_files, opts$network)

  log_msg("[INFO] Redes detectadas: %d", length(network_files), verbose = opts$verbose)
  log_msg("[INFO] Redes a procesar: %d", length(selected), verbose = opts$verbose)

  errors <- 0L
  for (network_file in selected) {
    ok <- TRUE
    tryCatch(
      {
        process_network(
          network_file = network_file,
          output_root = opts$output_dir,
          sample_sizes = opts$n_samples,
          seed = opts$seed,
          n_replications = opts$n_replications,
          max_targets = opts$max_targets,
          target_type = opts$target_type,
          min_class_balance = opts$min_class_balance,
          continuous_n_bins = opts$continuous_n_bins,
          continuous_n_bins_min = opts$continuous_n_bins_min,
          continuous_n_bins_max = opts$continuous_n_bins_max,
          continuous_min_bin_prop = opts$continuous_min_bin_prop,
          discretize_continuous_targets = opts$discretize_continuous_targets,
          balance_eval_samples = opts$balance_eval_samples,
          verbose = opts$verbose
        )
      },
      error = function(e) {
        ok <<- FALSE
        log_msg("[ERROR] Falló %s: %s", basename(network_file), conditionMessage(e), verbose = TRUE)
      }
    )
    if (!ok) errors <- errors + 1L
  }

  if (errors > 0L) {
    log_msg("[WARN] Finalizado con %d error(es).", errors, verbose = TRUE)
    quit(status = 1L)
  }

  log_msg("[INFO] Proceso finalizado correctamente.", verbose = opts$verbose)
}

if (sys.nframe() == 0) {
  main()
}
