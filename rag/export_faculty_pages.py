import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def _sanitize_filename_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.netloc or "unknown").replace(":", "_")
        path = p.path.strip("/").replace("/", "_")
        name = f"{host}_{path}" if path else host
        # remove characters invalid on Windows
        name = re.sub(r'[^A-Za-z0-9._-]+', "_", name)
        return name or "page"
    except Exception:
        base = re.sub(r'[^A-Za-z0-9._-]+', "_", url)
        return base[:100] or "page"


def _get(obj: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    # try case-insensitive
    lower_map = {str(k).lower(): v for k, v in obj.items()}
    for k in keys:
        v = lower_map.get(k.lower())
        if v not in (None, ""):
            return v
    return None


FACULTY_LIST_KEYS = [
    "faculty_members", "faculty", "members", "people", "staff", "profiles"
]
PAGE_URL_KEYS = ["faculty_page_url", "page_url", "url", "source_url"]


def _iter_pages(root: Any) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
    """
    Recursively yield (page_url, faculty_list) pairs.
    """
    if isinstance(root, dict):
        # try to find a faculty list
        faculty_list = None
        for key in FACULTY_LIST_KEYS:
            val = root.get(key)
            if isinstance(val, list) and val:
                faculty_list = val
                break
        page_url = None
        for key in PAGE_URL_KEYS:
            val = root.get(key)
            if isinstance(val, str) and val.startswith("http"):
                page_url = val
                break

        if faculty_list and page_url:
            yield page_url, faculty_list

        # recurse into children
        for v in root.values():
            yield from _iter_pages(v)
    elif isinstance(root, list):
        for v in root:
            yield from _iter_pages(v)


def _format_faculty_line(member: Dict[str, Any]) -> str:
    name = _get(member, "name", "full_name", "faculty_name") or ""
    designation = _get(member, "designation", "title", "role") or ""
    department = _get(member, "department", "dept", "school") or ""
    fid = _get(member, "id", "employee_id", "emp_id", "faculty_id") or ""
    intercom = _get(member, "intercom", "phone_ext", "extension") or ""
    email = _get(member, "email", "e_mail", "mail") or ""
    qualification = _get(member, "qualification", "qualifications", "education") or ""
    research_areas = _get(member, "research_areas", "research areas", "research", "areas_of_research") or ""
    specialization = _get(member, "specialization", "specializations", "area_of_specialization") or ""
    scholar_id = _get(member, "google_scholar_id", "googleScholarId", "scholar_id") or ""

    # Ensure strings
    to_str = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    parts: list[str] = []

    if name:
        parts.append(f"name \"{to_str(name)}\"")
    if designation:
        parts.append(f"has designation \"{to_str(designation)}\"")
    if department:
        parts.append(f"in department \"{to_str(department)}\"")
    if fid:
        parts.append(f"with id \"{to_str(fid)}\"")
    if intercom:
        parts.append(f"and intercom \"{to_str(intercom)}\"")
    if email:
        parts.append(f"with email \"{to_str(email)}\"")
    if qualification:
        parts.append(f"and has qualification \"{to_str(qualification)}\"")
    if research_areas:
        parts.append(f"with research areas \"{to_str(research_areas)}\"")
    if specialization:
        parts.append(f"and specialization \"{to_str(specialization)}\"")
    if scholar_id:
        parts.append(f"with google_scholar_id \"{to_str(scholar_id)}\"")

    # Fallback if everything missing
    if not parts:
        return "name \"\""

    # Join with spaces; phrases already include conjunctions
    return " ".join(parts)


def export_faculty_pages(json_path: Path, out_dir: Path) -> int:
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = list(_iter_pages(data))
    if not pages:
        # fallback: if no explicit pages, write a single file from any list of members found
        # and name it from the json filename
        aggregated: List[Dict[str, Any]] = []
        for _, fac in _iter_pages({"faculty": data}):
            aggregated.extend(fac)
        if not aggregated:
            return 0
        out_file = out_dir / (json_path.stem + "_faculty.txt")
        with open(out_file, "w", encoding="utf-8") as out:
            for m in aggregated:
                out.write(_format_faculty_line(m) + "\n")
        return 1

    written = 0
    for page_url, faculty_list in pages:
        fname = _sanitize_filename_from_url(page_url) + ".txt"
        out_file = out_dir / fname
        with open(out_file, "w", encoding="utf-8") as out:
            for member in faculty_list:
                out.write(_format_faculty_line(member) + "\n")
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Export per-faculty-page structured TXT files from a faculty JSON.")
    parser.add_argument("--json", type=str, required=True, help="Path to faculty_data_complete.json")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write page TXT files")
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not json_path.exists():
        print(f"JSON not found: {json_path}")
        return
    n = export_faculty_pages(json_path, out_dir)
    print(f"Wrote {n} page files to {out_dir}")


if __name__ == "__main__":
    main()


