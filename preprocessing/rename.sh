#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [-n|--dry-run] <directory>" >&2
  echo "Recursively rename files and folders: replace '-' with '_' in names." >&2
}

dry_run=false
target_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$target_dir" ]]; then
        target_dir="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$target_dir" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$target_dir" ]]; then
  echo "Error: '$target_dir' is not a directory" >&2
  exit 1
fi

# Traverse depth-first so children are renamed before their parents
while IFS= read -r -d '' path; do
  base_name=$(basename "$path")
  new_base_name=${base_name//-/_}

  if [[ "$base_name" == "$new_base_name" ]]; then
    continue
  fi

  dir_name=$(dirname "$path")
  new_path="$dir_name/$new_base_name"

  if [[ -e "$new_path" ]]; then
    echo "Skipping: target already exists -> $new_path" >&2
    continue
  fi

  if $dry_run; then
    echo "Would rename: $path -> $new_path"
  else
    mv -v "$path" "$new_path"
  fi
done < <(find "$target_dir" -depth -name '*-*' -print0)

if $dry_run; then
  echo "Dry run complete. No changes were made."
fi


