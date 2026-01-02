import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
import pandas as pd

def fetch_cogatlas():
    """Fetch the entire ontology from cognitiveatlas.org."""

    base = "https://www.cognitiveatlas.org"
    cogatlas = {}
    sections = []

    # pages: ["a", "b", ..., "z"]
    for alpha in tqdm(range(ord('a'), ord('z') + 1), total=26):
        alpha = chr(alpha)
        url = base + f"/concepts/{alpha}"
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for section in soup.find_all("a", class_="concept tooltip"):
            # get all concept data
            url, concept = section.get("href"), section.get_text(strip=True)
            url = f"https://www.cognitiveatlas.org/concept/json/{url.split("/")[-1]}"
            concept = concept.strip()
            sections.append((concept, url))

    cogatlas = {}
    for concept, url in tqdm(sections, total=len(sections)):
        json = requests.get(url).json()
        json["name"] = json["name"].strip()
        concept = concept.strip()
        assert json["name"] == concept
        cogatlas[concept] = json

    return cogatlas

def fetch_cogatlas_tasks():
    """Fetch the entire ontology from cognitiveatlas.org."""

    base = "https://www.cognitiveatlas.org"
    cogatlas = {}
    sections = []

    # pages: ["a", "b", ..., "z"]
    for alpha in tqdm(range(ord('a'), ord('z') + 1), total=26):
        alpha = chr(alpha)
        url = base + f"/tasks/{alpha}"
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for section in soup.find_all("a", class_="task tooltip"):
            # get all task data
            url, task = section.get("href"), section.get_text(strip=True)
            url = f"https://www.cognitiveatlas.org/task/json/{url.split("/")[-1]}"
            task = task.strip()
            sections.append((task, url))

    cogatlas = {}
    for task, url in tqdm(sections, total=len(sections)):
        json = requests.get(url).json()
        json["name"] = json["name"].strip()
        task = task.strip()
        assert json["name"] == task
        cogatlas[task] = json

    return cogatlas


def fetch_cogatlas_disorders():
    """Fetch disorder ontology from cognitiveatlas.org."""

    base = "https://www.cognitiveatlas.org"
    cogatlas = {}
    sections = []

    # Fetch the disorders page
    url = base + "/disorders"
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all disorder links (filter by href pattern to exclude traits/behaviors)
    for section in soup.find_all("a", class_="tooltip"):
        href = section.get("href")
        # Only get disorders (not traits or behaviors)
        if href and href.startswith("/disorder/id/"):
            disorder = section.get_text(strip=True)
            # Use the full disorder page URL, not a JSON endpoint
            disorder_url = f"{base}{href}"
            sections.append((disorder, disorder_url))

    # Fetch data for each disorder by scraping HTML pages
    cogatlas = {}
    for disorder, url in tqdm(sections, total=len(sections)):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract disorder information from the HTML page
            disorder_data = {
                "name": disorder.strip(),
                "id": url.split("/")[-2] if url.endswith("/") else url.split("/")[-1],
                "url": url,
                "definition_text": "",
                "alias": ""
            }

            # Try to find definition text
            definition_div = soup.find("div", class_="defn")
            if definition_div:
                disorder_data["definition_text"] = definition_div.get_text(strip=True)

            # Try to find alias
            alias_div = soup.find("div", class_="alias")
            if alias_div:
                disorder_data["alias"] = alias_div.get_text(strip=True)

            cogatlas[disorder] = disorder_data
        except requests.exceptions.HTTPError as e:
            print(f"Warning: Skipping '{disorder}' - HTTP error: {e}")
            continue
        except Exception as e:
            print(f"Warning: Skipping '{disorder}' - error: {e}")
            continue

    return cogatlas

def parse_cogatlas(cogatlas_meta):
    """Extract term/definition and graph dataframes."""
    # graph
    parents = []
    children = []
    relationship = []
    for k in cogatlas_meta.keys():
        for i in cogatlas_meta[k]['relationships']:
            if i["direction"] == "child":
                parents.append(k)
                children.append(i["name"])
                relationship.append(i["relationship"])

    df_graph = pd.DataFrame({"parent": parents, "child": children, "relationship": relationship})
    df_graph["parent"] = df_graph["parent"].str.strip()
    df_graph["child"] = df_graph["child"].str.strip()

    # extract a simplified df
    cogatlas = dict(
        term=[],
        definition=[],
        alias=[]
    )

    for k in cogatlas_meta.keys():
        cogatlas["term"].append(k)
        cogatlas["definition"].append(
            cogatlas_meta[k]["definition_text"].strip()
        )

        if "alias" not in cogatlas_meta[k].keys():
            cogatlas["alias"].append(None)
        else:
            alias = cogatlas_meta[k]["alias"].strip()
            cogatlas["alias"].append(None if alias == "" else alias)

    # filter
    df_cog = pd.DataFrame(cogatlas)
    mask = df_cog["definition"] != "WRITE DEFINITION HERE"
    df_no_def = df_cog[~mask]
    df_cog = df_cog[mask]
    df_graph = df_graph[~(df_graph["parent"].isin(df_no_def["term"]) | df_graph["child"].isin(df_no_def["term"]))]

    # checks
    assert df_graph["parent"].isin(df_cog["term"]).all()
    assert df_graph["child"].isin(df_cog["term"]).all()

    # ensure definitions start with capital letter and end with period
    df_cog["definition"] = df_cog["definition"].str.capitalize()
    m = ~df_cog["definition"].str.endswith(".")
    df_cog.loc[m, "definition"] = df_cog.loc[m, "definition"] + "."

    # save to pwd
    df_graph.to_parquet("cogatlas_graph.parquet")
    df_cog.to_parquet("cogatlas.parquet")

    return df_cog, df_graph

def parse_cogatlas_disorder(cogatlas_disorder_meta):
    """Extract disorder/definition dataframe."""
    # extract a simplified df
    cogatlas_disorder = dict(
        term=[],
        definition=[],
        alias=[]
    )

    for k in cogatlas_disorder_meta.keys():
        cogatlas_disorder["term"].append(k)

        # Handle missing definition_text
        if "definition_text" not in cogatlas_disorder_meta[k].keys() or cogatlas_disorder_meta[k]["definition_text"] == "":
            cogatlas_disorder["definition"].append("WRITE DEFINITION HERE")
        else:
            cogatlas_disorder["definition"].append(
                cogatlas_disorder_meta[k]["definition_text"].strip()
            )

        if "alias" not in cogatlas_disorder_meta[k].keys() or cogatlas_disorder_meta[k]["alias"] == "":
            cogatlas_disorder["alias"].append(None)
        else:
            alias = cogatlas_disorder_meta[k]["alias"].strip()
            cogatlas_disorder["alias"].append(None if alias == "" else alias)

    # filter
    df_disorder = pd.DataFrame(cogatlas_disorder)
    mask = df_disorder["definition"] != "WRITE DEFINITION HERE"
    df_no_def = df_disorder[~mask]
    df_disorder = df_disorder[mask]

    # ensure definitions start with capital letter and end with period
    df_disorder["definition"] = df_disorder["definition"].str.capitalize()
    m = ~df_disorder["definition"].str.endswith(".")
    df_disorder.loc[m, "definition"] = df_disorder.loc[m, "definition"] + "."

    # save to pwd
    df_disorder.to_parquet("cogatlas_disorder.parquet")

    return df_disorder



def parse_cogatlas_task(cogatlas_task_meta):
    """Extract task/definition and graph dataframes for tasks."""
    # graph - tasks have conditions and concepts as relationships
    tasks = []
    related_items = []
    relationship_types = []
    item_types = []

    for k in cogatlas_task_meta.keys():
        # Process conditions
        if 'conditions' in cogatlas_task_meta[k]:
            for condition in cogatlas_task_meta[k]['conditions']:
                if 'relationship' in condition and 'name' in condition:
                    tasks.append(k)
                    related_items.append(condition['name'])
                    relationship_types.append(condition['relationship'])
                    item_types.append('condition')

        # Process concepts
        if 'concepts' in cogatlas_task_meta[k]:
            for concept in cogatlas_task_meta[k]['concepts']:
                if 'relationship' in concept and 'name' in concept:
                    tasks.append(k)
                    related_items.append(concept['name'])
                    relationship_types.append(concept['relationship'])
                    item_types.append('concept')

    df_graph = pd.DataFrame({
        "task": tasks,
        "related_item": related_items,
        "relationship": relationship_types,
        "item_type": item_types
    })
    df_graph["task"] = df_graph["task"].str.strip()
    df_graph["related_item"] = df_graph["related_item"].str.strip()

    # extract a simplified df
    cogatlas_task = dict(
        term=[],
        definition=[],
        alias=[]
    )

    for k in cogatlas_task_meta.keys():
        cogatlas_task["term"].append(k)

        # Handle missing definition_text
        if "definition_text" not in cogatlas_task_meta[k].keys():
            cogatlas_task["definition"].append("WRITE DEFINITION HERE")
        else:
            cogatlas_task["definition"].append(
                cogatlas_task_meta[k]["definition_text"].strip()
            )

        if "alias" not in cogatlas_task_meta[k].keys():
            cogatlas_task["alias"].append(None)
        else:
            alias = cogatlas_task_meta[k]["alias"].strip()
            cogatlas_task["alias"].append(None if alias == "" else alias)

    # filter
    df_task = pd.DataFrame(cogatlas_task)
    mask = df_task["definition"] != "WRITE DEFINITION HERE"
    df_no_def = df_task[~mask]
    df_task = df_task[mask]
    df_graph = df_graph[~df_graph["task"].isin(df_no_def["term"])]

    # checks
    assert df_graph["task"].isin(df_task["term"]).all()

    # ensure definitions start with capital letter and end with period
    df_task["definition"] = df_task["definition"].str.capitalize()
    m = ~df_task["definition"].str.endswith(".")
    df_task.loc[m, "definition"] = df_task.loc[m, "definition"] + "."

    # save to pwd
    df_graph.to_parquet("cogatlas_task_graph.parquet")
    df_task.to_parquet("cogatlas_task.parquet")

    return df_task, df_graph