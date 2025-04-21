use strict;
use Carp;
use Data::Dumper;
use File::Temp;
use File::Slurp;
use File::Basename;
use IPC::Run 'run';
use JSON;
use Bio::KBase::AppService::AppConfig;
use Bio::KBase::AppService::AppScript;
use Cwd;

our $global_ws;
our $global_token;

#my $script = Bio::KBase::AppService::AppScript->new(\&create_rag_db, \&preflight);
my $script = Bio::KBase::AppService::AppScript->new(\&create_rag_db);
my $rc = $script->run(\@ARGV);
exit $rc;

sub create_rag_db {
    my ($app, $app_def, $raw_params, $params) = @_;
    
    my $script = "";
    my @cmd = ($script); 
    # TODO: add commands

    my $ok = run(\@cmd);
}
