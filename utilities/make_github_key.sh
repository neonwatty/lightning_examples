#!/bin/bash

# Set default key name
EMAIL="jermwatt@gmail.com"
KEY_NAME="github_ssh_key"

# Generate a new SSH key pair
ssh-keygen -t ed25519 -C $EMAIL -f ~/.ssh/$KEY_NAME -N ""

# Add the key to the SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/$KEY_NAME

# Print the public key
echo "Your new SSH public key (add this to GitHub):"
cat ~/.ssh/$KEY_NAME.pub