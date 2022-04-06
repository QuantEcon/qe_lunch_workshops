# Github Codespaces

### Setup and usage of Github Codespaces

Aakash Gupta Choudhury

QuantEcon lunch, March 22, 2022

## What is a Codespace?

A codespace is a development environment that's hosted in the cloud. It is instant and meant to be ephemeral. Codespaces support configuration files, which enables 
everyone using the project to have repeatable codespace configuration and development environment. Users can choose between 2 cores to 32 core machines, hosted in the cloud. The machines are VM-based and have a container with the code base running on top of it.

## Enabling Github Codespaces for your project

- Github organization account should either be using [Github Team](https://github.com/team) or [Github Enterprise Cloud](https://github.com/enterprise).
- Enable Codespaces from the settings for members of the organization, and set a spending limit to activate it.

## Accessing Codespaces

- A new codespace can be created from any branch, PR, commit of the repository in your browser.
- Can also be accessed using Github CLI in your local machine, using `gh codespace`. This uses `ssh` to connect to the VM. 
- Can be accessed with the Visual Studio Code in your local machine, by using `Github Codespaces` extension. 

## Configuring Codespaces

- To customize the environemnt used by codespaces, `devcontainer.json` or Dockerfile can be used. 
- Customizations can be done on a per-project and per-branch basis. 

## Port forwarding

- Gives you access to TCP ports running within your codespace.
- Enables you to access the application from the browser in you local machine to test and debug it. 
- Enables you to share the port within your organization or publicly.
