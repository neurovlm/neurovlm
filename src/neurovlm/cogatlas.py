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