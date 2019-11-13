require 'torch'
require 'nn'

require 'LanguageModel'


-- version of sample which passes in a coroutine to mess with 
-- the probability weights


-- try doing the punctuation comparison by index rather than token
-- I think it's a wide character issue

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'Projects/Musketeers/Musketeers2/Musketeers2_cp_176000.t7')
cmd:option('-vocab', 'Projects/Musketeers/anatomy_of_melancholy.txt')
--cmd:option('-notpunct', '’')
cmd:option('-notpunct', '')
cmd:option('-suppress', '')
cmd:option('-alliterate', '')
cmd:option('-length', 5000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', .3)
cmd:option('-verbose', 0)


local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local tokens = {}
local punctuation = {}


local msg
if opt.verbose == 1 then print(msg) end

local wmap = {}

print("Loading vocab from " .. opt.vocab)
local f = io.open(opt.vocab, "r")
local text = f:read("*all")
local suppressPat = nil
if #opt.suppress > 0 then
  suppressPat = '[' .. opt.suppress .. ']'
end
for w in string.gmatch(text, "%S+") do
  if suppressPat then
    if not w:find(suppressPat) then
      wmap[w] = 1
    end
  else
    wmap[w] = 1
  end
end

local words = {}

for w, _ in pairs(wmap) do
  words[ #words + 1 ] = w
end

print(#words)


local punct2 = " \n.:;,“”-()_"


function get_matches(ws)
  local matches = {}
  for p, _ in pairs(punctuation) do
    matches[p] = 1
  end
  if ws then
    for _, w in pairs(ws) do
      local idx = model.token_to_idx[w:sub(1,1)]
      if idx ~= nil then
        matches[idx] = 1
      else
        --print("unknown token: " .. w:sub(1,1))
      end
    end
  end
  return matches
end

function matches_to_weights(matches)
  local weights = {}
  for idx, _ in pairs(tokens) do
    if matches[idx] ~= nil then
      weights[idx] = 1
    else
      weights[idx] = 0
    end
  end
  return weights
end

function init_vocab(ws)
  local v = {}
  for i, w in pairs(ws) do
    v[i] = w
  end
  return v
end

function prune_vocab(ov, next_char)
  local v = {}
  for _, w in pairs(ov) do
    if next_char == w:sub(1, 1) then
      v[#v+1] = w:sub(2, #w)
    end
  end
  return v
end


tuner = coroutine.create(function(prev_char)
  local vocab = init_vocab(words)
  local current_word = ''
  while true
    do
      local weights = {}
      local matches = get_matches(vocab)
      local weights = matches_to_weights(matches)
      p = coroutine.yield(weights)
      local next_idx = p[{1,1}]
      local next_char = model.idx_to_token[next_idx]
      if current_word == 'd' and next_char == ' ' then
        print(matches)
      end
      if punctuation[next_idx] then
        vocab = init_vocab(words)
        print(current_word)
        print(next_char)
        current_word = ''
      else
        current_word = current_word .. next_char
        vocab = prune_vocab(vocab, next_char)
        if #vocab < 1 then
          vocab = init_vocab(words)
        end
      end
    end
end)


function make_alliterate(char)
  return coroutine.create(function(prev_char)
    local first_t = {}
    first_t[char] = 1
    first_t[char:upper()] = 1
    first_t[' '] = 1
    local weight_t = matches_to_weights(first_t)
    local weights = weight_t
    while true
      do
        p = coroutine.yield(weights)
        local next_char = model.idx_to_token[p[{1,1}]]
        if next_char:match("%W") then
          weights = weight_t
        else
          weights = {}
        end
      end
  end)
end

model:evaluate()

for idx, token in pairs(model.idx_to_token) do
  tokens[idx] = 1
  if token:match('%W') and not opt.notpunct:find("%" .. token) then
    punctuation[idx] = 1
  end
end

print("Tokens")



print(tokens, punctuation)




-- local mod = tuner

-- if opt.alliterate then
--   mod = make_alliterate(opt.alliterate:sub(1,1))
-- end

local sample = model:sample_hacked(opt, tuner)
print(sample)
