//! Site identity for `graph migrate` / `refresh` / `check`.
//!
//! Read from the site's `config.toml` `[extra.graph]`, falling back to
//! `title` / `description`. The zola binary does not name a product.

use std::path::Path;

use errors::{anyhow, Result};
use toml::Value;

use super::schema::{GraphStore, Organization};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphSiteConfig {
    pub title: String,
    pub description: String,
    pub org_id: String,
    pub org_name: String,
    pub org_logo: String,
    pub pillars: Vec<String>,
    pub no_related: Vec<(String, String)>,
    pub required_redirects: Vec<(String, String)>,
    pub overview_instruction: Option<String>,
}

impl GraphSiteConfig {
    pub fn load(config_file: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(config_file).map_err(|e| anyhow!("read config: {e}"))?;
        let cfg: Value = toml::from_str(&text).map_err(|e| anyhow!("parse config: {e}"))?;
        Ok(Self::from_toml(&cfg))
    }

    pub fn load_optional(config_file: &Path) -> Self {
        Self::load(config_file).unwrap_or_default()
    }

    pub fn from_toml(cfg: &Value) -> Self {
        let title = cfg
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let description = cfg
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let g = cfg.get("extra").and_then(|e| e.get("graph"));
        let org_name = str_field(g, "org_name").unwrap_or_else(|| title.clone());
        let org_id = str_field(g, "org_id").unwrap_or_else(|| {
            if org_name.is_empty() {
                String::new()
            } else {
                format!("org:{}", slug(&org_name))
            }
        });
        Self {
            org_logo: str_field(g, "org_logo").unwrap_or_default(),
            org_id,
            org_name,
            pillars: str_list(g, "pillars"),
            no_related: pair_list(g, "no_related"),
            required_redirects: pair_list(g, "required_redirects"),
            overview_instruction: str_field(g, "overview_instruction"),
            title,
            description,
        }
    }

    pub fn organization(&self) -> Option<Organization> {
        if self.org_name.is_empty() {
            return None;
        }
        Some(Organization {
            id: if self.org_id.is_empty() {
                format!("org:{}", slug(&self.org_name))
            } else {
                self.org_id.clone()
            },
            name: self.org_name.clone(),
            url: String::new(),
            logo: self.org_logo.clone(),
            same_as: vec![],
        })
    }

    pub fn is_pillar(&self, id: &str) -> bool {
        self.pillars.iter().any(|p| p == id)
    }
}

pub fn seed_organization(store: &mut GraphStore, site: &GraphSiteConfig) {
    if !store.organizations.is_empty() {
        return;
    }
    if let Some(org) = site.organization() {
        store.organizations.push(org);
    }
}

fn str_field(g: Option<&Value>, key: &str) -> Option<String> {
    g.and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn str_list(g: Option<&Value>, key: &str) -> Vec<String> {
    g.and_then(|v| v.get(key))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn pair_list(g: Option<&Value>, key: &str) -> Vec<(String, String)> {
    g.and_then(|v| v.get(key))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let pair = v.as_array()?;
                    if pair.len() != 2 {
                        return None;
                    }
                    Some((
                        pair[0].as_str()?.trim().to_string(),
                        pair[1].as_str()?.trim().to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_graph_wins() {
        let cfg: Value = toml::from_str(
            r#"
title = "Ignored Title"
[extra.graph]
org_id = "org:acme"
org_name = "Acme"
org_logo = "/logo.webp"
pillars = ["content/_index.md"]
no_related = [["content/_index.md", "content/other/index.md"]]
required_redirects = [["/a/", "/b/"]]
"#,
        )
        .unwrap();
        let s = GraphSiteConfig::from_toml(&cfg);
        let org = s.organization().unwrap();
        assert_eq!(org.id, "org:acme");
        assert_eq!(org.name, "Acme");
        assert_eq!(org.logo, "/logo.webp");
        assert!(s.is_pillar("content/_index.md"));
        assert_eq!(s.no_related[0].1, "content/other/index.md");
        assert_eq!(s.required_redirects[0], ("/a/".into(), "/b/".into()));
    }

    #[test]
    fn title_fallback_slugs_org_id() {
        let cfg: Value = toml::from_str("title = \"Acme Labs\"\n").unwrap();
        let s = GraphSiteConfig::from_toml(&cfg);
        let org = s.organization().unwrap();
        assert_eq!(org.id, "org:acme-labs");
        assert_eq!(org.name, "Acme Labs");
        assert!(s.pillars.is_empty());
        assert!(s.no_related.is_empty());
    }

    #[test]
    fn empty_config_seeds_nothing() {
        let s = GraphSiteConfig::from_toml(&toml::from_str("base_url = \"https://x/\"\n").unwrap());
        assert!(s.organization().is_none());
    }
}
